#!/usr/bin/env python
# encoding: utf-8
"""
@file im.py
@brief Image manipulation routines

@package libtim.im
@brief Image manipulation routines
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120403

Image manipulation functions.
"""

#=============================================================================
# Import libraries here
#=============================================================================

import os, sys
import getpass
import pyfits
import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
import datetime

from file import filenamify
from util import mkfitshdr

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def mk_rad_mask(r0, r1=None, norm=True, center=None):
	"""
	Make a rectangular matrix where the value of each element is the distance to the center to the shape **r0** and **r1**. I.e. the center edge has value 1, the corners have value sqrt(2) in case of a square matrix.

	If only r0 is given, the matrix will be (r0, r0). If ry is also given, the matrix will be (r0, r1)

	To convert this to a circular binary mask, use mk_rad_mask(r0) < 1

	@param [in] r0 The width (and height if r1==None) of the mask.
	@param [in] r1 The height of the mask.
	@param [in] norm Normalize the distance such that 2/(r0, r1) equals a distance of 1.
	@param [in] center Set distance origin to **center** (defaults to the middle pixel of the rectangle)
	"""

	if (not r1):
		r1 = r0
	if (r0 < 1 or r1 < 1):
		raise ValueError("r0, r1 should be > 0")
	
	if (not center):
		center = (r0/2.0, r1/2.0)

	# N.B. These are calculated separately because we cannot calculate  
	# 2.0/r0 first and multiply r0v with it depending on **norm**, this will 
	# yield different results due to rounding errors.
	if (norm):
		r0v = np.linspace(-1, 1, r0).reshape(-1,1)
		r1v = np.linspace(-1, 1, r1).reshape(1,-1)
	else:
		r0v = np.linspace(-r0/2.0, r0/2.0, r0).reshape(-1,1)
		r1v = np.linspace(-r1/2.0, r1/2.0, r1).reshape(1,-1)
	
	return (r0v**2. + r1v**2.)**0.5

def mk_rad_prof(data, maxrange=None, step=1, procf=np.mean):
	"""
	Make mean radial profile of **data**.

	Make a radial profile of **data** up to **maxrange** pixels from the center

	@param [in] data 2D array of data
	@param [in] maxrange Range of profile to make in pixels (min(data.shape)/2 if None)
	@param [in] step Stepsize in pixels (has to be >=1)
	@param [in] procf Function to apply to each annulus (np.mean
	@return Radial profile binned per pixel as numpy.ndarray.
	"""

	# Set maxrange if not set
	if (not maxrange):
		maxrange = min(data.shape)/2

	# Make radial mask
	rad_mask = mk_rad_mask(data.shape[0], data.shape[1], norm=False, center=None)
	# Above code is identical to
	# rad_mask = np.indices(data.shape) - (np.r_[data.shape]/2).reshape(-1,1,1)
	# rad_mask = np.sqrt((rad_mask**2.0).sum(0))


	# Make radial profile using <step> pixel wide annuli with increasing 
	# radius until <maxrange>. Calculate the mean in each annulus
	profile = [procf(data[(rad_mask >= i) & (rad_mask < i+step)]) for i in xrange(0, maxrange, step)]

	# Code above is equivalent to:
	#profile = []
	#for i in xrange(0, maxrange, step):
	#	this_mask = (rad_mask >= i) & (rad_mask < i+step)
	#	profile.append(procf(data[this_mask]))

	return np.r_[profile]

def df_corr(indata, flatfield=None, darkfield=None, darkfac=[1.0, 1.0], thresh=0, copy=True):
	"""
	Correct 2-d array **data** with **flatfield** and **darkfield**.

	Use **darkfac** to scale the darkfield in case of incorrect exposure.

	**thresh** can be used to mask the flatfield image: intensity lower than this factor times the maximum intensity will not be flatfielded.

	Final returned data will be:

	\code
	(data - darkfac[0] * darkfield) / (flatfield - darkfac[1] * darkfield)
	\endcode

	or a part of that in case only dark- or flatfield is given.

	@param [in] indata Image to flat/dark-field
	@param [in] flatfield Flatfield image to use
	@param [in] darkfield Darkfield image to use
	@param [in] darkfac Correction factors for darkfield to use if exposure is mismatched
	@param [in] thresh Threshold to use for flatfielding
	@param [in] copy Non-destructive: copy input before correcting
	@return Corrected image as numpy.ndarray
	"""
	
	# If no FF and DF, return original data
	if (flatfield == None and darkfield == None):
		return indata
		
	if (copy):
		data = indata.copy()

	if (darkfield != None):
		# If we have a darkfield, remove it from the data, but clip at 0
		data -= darkfac[0] * darkfield
		data[data < 0] = 0
		# If we have a flatfield as well, dark-correct the flatfield and clip
		if (flatfield != None):
			flatfield -= darkfac[1] * darkfield
			flatfield[flatfield < 0] = 0

	# If we have a flatfield, correct the image, but only divide for nonzero flat
	if (flatfield != None):
		flatmask = (flatfield > thresh*flatfield.max())
		data[flatmask] /= flatfield[flatmask]

	return data

def store_2ddata(data, fname, pltitle='', dir='./', fits=False, plot=True, plrange=(None, None), log=False, rollaxes=False, cmap='RdYlBu', xlab='X [pix]', ylab='Y [pix]', cbarlab=None, hdr=(), ident=True):
	"""
	Store **data** to disk as FITS and/or plot as annotated plot in PDF.

	@param [in] data 2D data array to show
	@param [in] fname Filename base to use (also fallback for plot title)
	@param [in] pltitle Plot title (if given)
	@param [in] dir Output directory
	@param [in] fits Toggle FITS output
	@param [in] plot Toggle 2D plot output as PDF
	@param [in] plrange Use this range for plotting in imshow() (None for autoscale)
	@param [in] log Take logarithm of data before storing.
	@param [in] rollaxes Roll axes for PDF plot such that (0,0) is the center
	@param [in] cmap Colormap to use for PDF
	@param [in] xlab X-axis label
	@param [in] ylab Y-axis label
	@param [in] cbarlab Colorbar label (for units)
	@param [in] hdr Additional FITS header items, give a list of tuples: [(key1, val1), (key2, val2)]
	@param [in] ident Add identification string to plots
	@returns Tuple of (fitsfile path, plotfile path)
	"""

	# Do not store empty data
	if (len(data) <= 0):
		return

	data_arr = np.asanyarray(data)
	if (log):
		data_arr = np.log10(data_arr)
	extent = None
	if (rollaxes):
		sh = data_arr.shape
		extent = (-sh[1]/2., sh[1]/2., -sh[0]/2., sh[0]/2.)

	# Check if dir exists, or create
	if (not os.path.isdir(dir)):
		os.makedirs(dir)

	fitsfile = filenamify(fname)+'.fits'
	fitspath = os.path.join(dir, fitsfile)
	plotfile = filenamify(fname)+'.pdf'
	plotpath = os.path.join(dir, plotfile)

	if (fits):
		# Generate some metadata. Also store plot settings here
		hdr_dict = dict({'filename':fitsfile, 
				'desc':fname, 
				'title':pltitle,
				'plxlab': xlab,
				'plylab': ylab,
				'pllog': log,
				'plrlxs': rollaxes,
				'plcmap': cmap,
				'plrng0': plrange[0] if plrange[0] else 0,
				'plrng1': plrange[1] if plrange[1] else 0,
				}.items()
				+ dict(hdr).items())
		hdr = mkfitshdr(hdr_dict)
		# Store data to disk
		pyfits.writeto(fitspath, data_arr, header=hdr, clobber=True, checksum=True)

	if (plot):
		#plot_from_fits(fitspath)
		pltit = fname
		if (pltitle):
			pltit = pltitle

		# Plot without GUI, using matplotlib internals
		fig = Figure(figsize=(6,6))
		ax = fig.add_subplot(111)
		# Make margin smaller
		fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
		img=0
		# Colormaps
		# plus min: cmap=cm.get_cmap('RdYlBu')
		# linear: cmap=cm.get_cmap('YlOrBr')
		# gray: cmap=cm.get_cmap('gray')
		img = ax.imshow(data_arr, interpolation='nearest', cmap=cm.get_cmap(cmap), aspect='equal', extent=extent, vmin=plrange[0], vmax=plrange[1])
		ax.set_title(pltit)
		ax.set_xlabel(xlab)
		ax.set_ylabel(ylab)
		# dimension 0 is height, dimension 1 is width
		# When the width is equal or more than the height, use a horizontal bar, otherwise use vertical
		if (data_arr.shape[0]/data_arr.shape[1] >= 1.0):
			cbar = fig.colorbar(img, orientation='vertical', aspect=30, pad=0.05, shrink=0.8)
		else:
			cbar = fig.colorbar(img, orientation='horizontal', aspect=30, pad=0.12, shrink=0.8)
		if (cbarlab):
			cbar.set_label(cbarlab)

		# Add ID string
		if (ident):
			# Make ID string
			datestr = datetime.datetime.utcnow().isoformat()+'Z'
			# Put this in try-except because os.getlogin() fails in screen(1)
			try:
				idstr = "%s@%s %s %s" % (os.getlogin(), os.uname()[1], datestr, sys.argv[0])
			except OSError:
				idstr = "%s@%s %s %s" % (getpass.getuser(), os.uname()[1], datestr, sys.argv[0])

			ax.text(0.01, 0.01, idstr, fontsize=7, transform=fig.transFigure)

		canvas = FigureCanvas(fig)
		#canvas.print_figure(plotfile, bbox_inches='tight')
		canvas.print_figure(plotpath)

	return (fitspath, plotpath)

def inter_imshow(data, desc="", doshow=True, dowait=True, log=False, rollaxes=False, cmap='RdYlBu', figid=None, **kwargs):
	"""Show data using matplotlib.imshow if **doshow** is true.

	Additionally, print **desc** just before plotting so users know what they see. If **dowait** is True (default), wait for input before continuing.

	This function is used to show intermediate results of analysis programs conditionally, i.e. only when a certain flag is set.

	@param [in] data 2D data to plot
	@param [in] desc Text to print before plot and plot title
	@param [in] doshow Show only if this is True
	@param [in] dowait If set, wait before continuing
	@param [in] log Take logarithm of data before plotting
	@param [in] rollaxes Roll axes for plot such that (0,0) is the center
	@param [in] cmap Colormap to use (RdYlBu or YlOrBr are nice)
	@param [in] figid Figure id to use in plt.figure(figid), can be used to re-use plot windows.
	@param [in] **kwargs Additional arguments passed to imshow()
	"""

	if (not doshow):
		return

	import pylab as plt
	print "inter_imshow(): " + desc

	# Pre-format data
	data_arr = np.asanyarray(data)
	if (log):
		data_arr = np.log10(data_arr)

	# Check if we want to roll the axis
	extent = None
	if (rollaxes):
		sh = data_arr.shape
		extent = (-sh[1]/2., sh[1]/2., -sh[0]/2., sh[0]/2.)

	fig = plt.figure(figid)
	fig.clf()
	ax = fig.add_subplot(111)
	fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	ax.set_title(desc)

	img = ax.imshow(data_arr, extent=extent, cmap=cm.get_cmap(cmap), **kwargs)
	fig.colorbar(img, aspect=30, pad=0.05)
	plt.draw()

	# If we want to wait, ask user for input, discard it and continue
	if (dowait):
		raw_input()
