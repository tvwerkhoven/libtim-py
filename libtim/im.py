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
import numpy as N
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
import unittest
import datetime

from file import filenamify
from util import mkfitshdr

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def mk_rad_mask(r0, r1=None):
	"""
	Make a rectangular matrix where the value of each element is the distance to the center normalized to the shape **r0** and **r1**. I.e. the center edge has value 1, the corners have value sqrt(2) in case of a square matrix.

	If only r0 is given, the matrix will be (r0, r0). If ry is also given, the matrix will be (r0, r1)

	To convert this to a circular binary mask, use mk_rad_mask(r0) < 1

	@param [in] r0 The width (and height if r1==None) of the mask.
	@param [in] r1 The height of the mask.
	"""

	if (not r1):
		r1 = r0
	if (r0 < 1 or r1 < 1):
		raise ValueError("r0, r1 should be > 0")

	r0v = ((N.arange(1.0*r0) - 0.5*r0)*2/r0).reshape(-1,1)
	r1v = ((N.arange(1.0*r1) - 0.5*r1)*2/r1).reshape(1,-1)
	grid_rad = (r0v**2. + r1v**2.)**0.5
	return grid_rad

def mk_rad_prof(data, maxrange=None):
	"""
	Make radial profile of **data**.

	Make a radial profile of **data** up to **maxrange** pixels from the center or min(data.shape)/2 if not set.

	@param [in] data 2D array of data
	@param [in] maxrange Range of profile to make in pixels (min(data.shape)/2 if None)
	@return Radial profile binned per pixel as numpy.ndarray.
	"""

	step = 1

	# Set maxrange if not set
	if (not maxrange):
		maxrange = min(data.shape)/2

	# Make radial mask
	rad_mask = N.indices(data.shape) - (N.r_[data.shape]/2).reshape(-1,1,1)
	rad_mask = N.sqrt((rad_mask**2.0).sum(0))

	# Init radial intensity profile
	profile = []

	# Make radial profile using <step> pixel wide annuli with increasing radius.
	for i in xrange(0, maxrange, step):
		this_mask = (rad_mask >= i) & (rad_mask < i+step)
		profile.append(data[this_mask].mean())

	return N.r_[profile]

def df_corr(data, flatfield=None, darkfield=None, darkfac=[1.0, 1.0], thresh=0):
	"""
	Correct 2-d array **data** with **flatfield** and **darkfield**.

	Use **darkfac** to scale the darkfield in case of incorrect exposure.

	**thresh** can be used to mask the flatfield image: intensity lower than this factor times the maximum intensity will not be flatfielded.

	Final returned data will be:

	\code
	(data - darkfac[0] * darkfield) / (flatfield - darkfac[1] * darkfield)
	\endcode

	or a part of that in case only dark- or flatfield is given.

	@param [in] data Image to flat/dark-field
	@param [in] flatfield Flatfield image to use
	@param [in] darkfield Darkfield image to use
	@param [in] darkfac Correction factors for darkfield to use if exposure is mismatched
	@param [in] thresh Threshold to use for flatfielding
	@return Corrected image as numpy.ndarray
	"""

	# If no FF and DF, return original data
	if (flatfield == None and darkfield == None):
		return data

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

def store_2ddata(data, fname, pltitle='', dir='./', fits=False, plot=True, plrange=(None, None), log=False, rollaxes=False, cmap='RdYlBu', xlab='X [pix]', ylab='Y [pix]', hdr=(), ident=True):
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
	@param [in] hdr Additional FITS header items, give a list of tuples: [(key1, val1), (key2, val2)]
	@param [in] ident Add identification string to plots
	@returns Tuple of (fitsfile path, plotfile path)
	"""

	# Do not store empty data
	if (len(data) <= 0):
		return

	data_arr = N.asanyarray(data)
	if (log):
		data_arr = N.log10(data_arr)
	extent = None
	if (rollaxes):
		sh = data_arr.shape
		extent = (-sh[1]/2., sh[1]/2., -sh[0]/2., sh[0]/2.)

	# Check if dir exists, or create
	if (not os.path.isdir(dir)):
		os.makedirs(dir)

	fitsfile = filenamify(fname)+'.fits'
	plotfile = filenamify(fname)+'.pdf'

	if (fits):
		# Generate some metadata
		hdr_dict = dict({'filename':fitsfile, 'desc':fname, 'title':pltitle}.items() + dict(hdr).items())
		hdr = mkfitshdr(hdr_dict)
		# Store data to disk
		pyfits.writeto(os.path.join(dir, fitsfile), data_arr, header=hdr, clobber=True, checksum=True)

	if (plot):
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
			fig.colorbar(img, orientation='vertical', aspect=30, pad=0.05, shrink=0.8)
		else:
			fig.colorbar(img, orientation='horizontal', aspect=30, pad=0.12, shrink=0.8)

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
		canvas.print_figure(os.path.join(dir, plotfile))

	return (fitsfile, plotfile)

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
	data_arr = N.asanyarray(data)
	if (log):
		data_arr = N.log10(data_arr)

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

class TestUtilFuncs(unittest.TestCase):
	def setUp(self):
		self.data = N.random.random((128,256))

	def test0a_mkradmask(self):
		"""Test mk_rad_mask"""
		mk_rad_mask(1)
		mk_rad_mask(25.0)
		mk_rad_mask(128.0)
		mk_rad_mask(10, 64)

	def test0b_mkradmask(self):
		"""Test mk_rad_mask fails"""
		with self.assertRaises(ValueError):
			mk_rad_mask(0)
		with self.assertRaises(ValueError):
			mk_rad_mask(-1)
		with self.assertRaises(ValueError):
			mk_rad_mask(-10, -10)

class TestInterImshow(unittest.TestCase):
	def setUp(self):
		self.data = N.random.random((128,256))

	# inter_imshow(data, desc="", doshow=True, dowait=True, log=False, rollaxes=False, cmap='RdYlBu', **kwargs):
	def test0a_show(self):
		"""Test simple plot with default args"""
		inter_imshow(self.data, dowait=False)

	def test1a_show_desc(self):
		"""Test plot with desc"""
		inter_imshow(self.data, "hello world", dowait=False)

	def test1b_show_doshow(self):
		"""Test plot without doshow"""
		inter_imshow(self.data, "hello world", doshow=False, dowait=False)

	def test1c_show_log(self):
		"""Test plot with log"""
		inter_imshow(self.data, "hello world", log=True, dowait=False)

	def test1d_show_rollaxes(self):
		"""Test plot with rollaxes"""
		inter_imshow(self.data, "hello world", rollaxes=True, dowait=False)

	def test1e_show_cmap(self):
		"""Test plot with cmap"""
		inter_imshow(self.data, "hello world", cmap='YlOrBr', dowait=False)

class TestStoreData(unittest.TestCase):
	def setUp(self):
		self.im1 = N.random.random((640,480))

		axis = N.linspace(0,4*N.pi,320)
		self.im2 = 2.*N.sin(3*axis).reshape(1,-1) * 3.*N.cos(2*axis).reshape(-1,1)

	def test0a_show(self):
		"""Dummy test, show image"""
		plt.figure(0)
		plt.title('Image one, random rectangle')
		plt.imshow(self.im1)
		plt.colorbar()
		plt.figure(1)
		plt.title('Image two, wavy square')
		plt.imshow(self.im2)
		plt.colorbar()

		inter_imshow(self.im1, "random rectangle", cmap='YlOrBr', dowait=False)
		inter_imshow(self.im2, "wavy square", cmap='YlOrBr', dowait=False)

		raw_input()

	def test1a_store_error(self):
		"""Store images as FITS and PDF, check for errors"""
		fpaths = store_2ddata(self.im1, 'TestStoreData_im1', pltitle='Random image', dir='/tmp/', fits=True, plot=True, plrange=(None, None), log=False, rollaxes=True, cmap='RdYlBu', xlab='X [pix]', ylab='Y [pix]', hdr=[('author', 'TestStoreData')])

		if (fpaths[0]): os.remove(fpaths[0])
		if (fpaths[1]): os.remove(fpaths[1])

		fpaths = store_2ddata(self.im2, 'TestStoreData_im2', pltitle='Wavy image', dir='/tmp/', fits=True, plot=True, plrange=(0, 1), log=False, rollaxes=True, cmap='RdYlBu', xlab='X [pix]', ylab='Y [pix]', hdr=[('author', 'TestStoreData')])

		if (fpaths[0]): os.remove(fpaths[0])
		if (fpaths[1]): os.remove(fpaths[1])

	def test1a_store_filesize(self):
		"""Store images as FITS and PDF, check for filesize > 0"""

		for im, imname in zip([self.im1, self.im2], ['im1', 'im2']):
			fpaths = store_2ddata(im, 'TestStoreData_'+imname, pltitle='Random image', dir='/tmp/', fits=True, plot=True, plrange=(None, None), log=False, rollaxes=True, cmap='RdYlBu', xlab='X [pix]', ylab='Y [pix]', hdr=[('author', 'TestStoreData')])

			# Check existence
			self.assertTrue(os.path.isfile(fpaths[0]))
			self.assertTrue(os.path.isfile(fpaths[1]))

			# Check filesize
			self.assertGreater(os.path.getsize(fpaths[0]), 0)
			self.assertGreater(os.path.getsize(fpaths[1]), 0)

			# Remove files
			if (fpaths[0]): os.remove(fpaths[0])
			if (fpaths[1]): os.remove(fpaths[1])

class TestDarkFlatfield(unittest.TestCase):
	def setUp(self):
		"""Generate source image, darkfield, flatfield and simulated data"""
		self.sz = (257, 509)
		sz = self.sz
		grid = N.indices(sz)
		# Random darkfield
		self.dark = N.random.random(sz)*10
		# horizontal linear flatfield. 'src' is the real flatfield, 'flat' is the 'measured' flatfield with darkfield in it
		self.flatsrc = 1.0+grid[1]*25./sz[1]
		self.flat = 1.0+grid[1]*25./sz[1] + self.dark
		# Image is a sine/cosine shape
		self.src = 25 + 25.0*N.cos(grid[0]*12./sz[0])*N.sin(grid[1]*9./sz[1])
		# Simulated data is noisy
		self.data = self.src*self.flatsrc + self.dark

	# Shallow data tests
	def test0a_data(self):
		"""Dark, flat & data should be unequal"""
		self.assertFalse(N.allclose(self.dark, self.flat))
		self.assertFalse(N.allclose(self.src, self.flat))
		self.assertFalse(N.allclose(self.src, self.data))

	# Shallow function test
	def test1a_return_shape_type(self):
		"""Return type and shape should be the same"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in, self.flat, self.dark)
		self.assertEqual(self.data.dtype, test_corr.dtype)
		self.assertEqual(self.data.shape, test_corr.shape)

	def test1a_return_shape_type(self):
		"""Without dark and flat, return should be same as input"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in)
		self.assertTrue(N.allclose(test_corr, self.data))

	# Deep function test
	def test2a_df_check(self):
		"""Darkfield corrected data should subtract darkfield"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in, darkfield=self.dark)
		self.assertTrue(N.allclose(test_corr, self.src*self.flatsrc))

	def test2b_ff_check(self):
		"""Flatfield corrected data should divide by flatfield"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in, flatfield=self.flat)
		self.assertTrue(N.allclose(test_corr, self.data/self.flat))

	def test2c_df_ff_check(self):
		"""Dark/flat field corrected data should be same as src"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in, self.flat, self.dark)
		self.assertTrue(N.allclose(test_corr, self.src))

	def test2d_ff_zero_check(self):
		"""Flatfield of zeros or ones should do nothing"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in, flatfield=N.zeros(test_in.shape))
		self.assertTrue(N.allclose(test_corr, self.data))
		test_in = self.data.copy()
		test_corr = df_corr(test_in, flatfield=N.ones(test_in.shape))
		self.assertTrue(N.allclose(test_corr, self.data))

	def test2e_dark_zeros_check(self):
		"""Darkfield of zeros should do nothing"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in, darkfield=N.zeros(test_in.shape))
		self.assertTrue(N.allclose(test_corr, self.data))

	def test3a_high_dark(self):
		"""Check with high darkfield, should return zero array"""
		test_in = self.data.copy()
		test_df = N.ones(test_in.shape) * test_in.max()
		test_corr = df_corr(test_in, darkfield=test_df)
		self.assertEqual(N.sum(test_corr), 0)

	def test3b_ident_ff(self):
		"""Check with image itself as flatfield, should return ones"""
		test_in = self.data.copy()
		test_corr = df_corr(test_in, flatfield=test_in)
		self.assertTrue(N.allclose(test_corr, N.ones(test_corr.shape)))

	def test3c_random(self):
		"""Check with random data as input, flatfield and darkfield"""
		for iter in xrange(10):
			indat = N.round(N.random.random(self.sz)*100)
			ffdat = N.round(N.random.random(self.sz)*100)
			dfdat = N.round(N.random.random(self.sz)*100)
			df_corr(indat, flatfield=ffdat, darkfield=dfdat)

if __name__ == "__main__":
	import sys
	import pylab as plt
	sys.exit(unittest.main())
