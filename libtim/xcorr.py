#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
@file xcorr.py
@brief Measure image shift using cross-correlation

@package libtim.xcorr
@brief Measure image shift using cross-correlation
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120402

Measure image shifts using cross-correlation and other utilities.
"""

#==========================================================================
# Import libraries here
#==========================================================================

import numpy as np
import fft as _fft
import scipy.ndimage
import scipy.ndimage.fourier

#==========================================================================
# Defines
#==========================================================================

#==========================================================================
# Routines
#==========================================================================

def crosscorr(imlst, shrange, dsh=(1,1), refim=None):
	"""
	Cross-correlate images in **imlst**.

	**shrange** determines the range of shifts to cross-correlate for, and 
	**dsh** indicates the step-size to use, i.e. dsh = (2,2) means every 
	second shift distance is used.

	If **refim** is given, compare all images from **imlist** with this 
	image, otherwise cross-correlate all pairs of images in **imlist**.

	The returned data is a list of lists with correlation maps. The 
	correlation maps themselves have shape (2*shrange+1)/dsh. If refim is 
	given, the list of lists is Nx1 big, otherwise it is NxN big.

	Correlation maps are only square if shrange[0] == shrange[1] and dsh[0] == dsh[1].

	@param [in] imlst list of images to cross correlate
	@param [in] shrange shift range to calculate, format (sh0, sh1)
	@param [in] dsh shift range cadence
	@param [in] refim Reference image to cross correlate others against. If None, will use cross correlate all pairs from imlst
	@return NxN (refim==None) or Nx1 (refim!=None) list of cross-correlation maps. Each map is (2*shrange+1)/dsh big
	"""

	# Check if imlst is malformed
	try:
		test = imlst[0][0,0]*1
	except:
		raise ValueError("<imlst> should be a list of 2D arrays.")

	if (refim != None):
		if (refim.shape != imlst[0].shape):
			raise ValueError("<refim> should be same size as <imlst[0]>")

	# Convenience variables for shift range for two dimensions, 0 and 1
	sh0, sh1 = shrange
	dsh0, dsh1 = dsh
	imsz0, imsz1 = imlst[0].shape

	# Shift ranges need to be larger than 0
	if (sh0 < 0 or sh1 < 0):
		raise ValueError("<shrange> should be larger or equal to 0")

	# We need a crop window to allow for image shifting. If sh0 or sh1 is 0, 
	# use all data.
	sm_crop = [slice(sh0, -sh0) if sh0 else slice(None), 
		slice(sh1, -sh1) if sh1 else slice(None)]

	# Calculate correlation for all files with <refim> (output Nx1)
	if (refim != None):
		xcorr_mat = [[
			np.r_[ [[
			np.sum(refim[sh0+shi:imsz0-sh0+shi, sh1+shj:imsz1-sh1+shj] * fj[sm_crop])
			for shj in xrange(-sh1, sh1+1, dsh1)]
				for shi in xrange(-sh0, sh0+1, dsh0)]
				] # // np.r_
					for fj in imlst]]
	# Calculate correlation for all files with each other (output NxN)
	else:
		xcorr_mat = [[
			np.r_[ [[
			np.sum(fi[sh0+shi:imsz0-sh0+shi, sh1+shj:imsz1-sh1+shj] * fj[sm_crop])
			for shj in xrange(-sh1, sh1+1, dsh1)]
				for shi in xrange(-sh0, sh0+1, dsh0)]
				] # // np.r_
					for fj in imlst]
						for fi in imlst]

	return xcorr_mat

def crosscorr1(img, refim, shrange, dsh=(1,1)):
	"""
	Calculate cross-correlation for only 1 image. See crosscorr() for 
	details.
	"""

	if (img.shape != refim.shape):
		raise ValueError("<refim> should be same size as <img>")

	sh0, sh1 = shrange
	dsh0, dsh1 = dsh
	imsz0, imsz1 = img.shape

	# Shift ranges need to be larger than 0
	if (sh0 < 1 or sh1 < 1):
		raise ValueError("<shrange> should be larger than 0")

	# We need a crop window to allow for image shifting
	sm_crop = [slice(sh0, -sh0), slice(sh1, -sh1)]

	# Calculate correlation for img with <refim>
	xcorr = np.r_[ [[
		np.sum(refim[sh0+shi:imsz0-sh0+shi, sh1+shj:imsz1-sh1+shj] * img[sm_crop])
		for shj in xrange(-sh1, sh1+1, dsh1)]
			for shi in xrange(-sh0, sh0+1, dsh0)]
				] # // np.r_

	return xcorr

def plot_img_mat(img_mat, fignum=0, pause=True, pltit="", **kwargs):
	"""
	Plot grid of images in one figure.

	@param [in] img_mat A list of lists of 2D images
	@param [in] fignum pylab plot figure to use
	@param [in] pause Pause after showing or not
	@param [in] pltit Plot title
	@param [in] **kwargs additional arguments for pylab.imshow
	"""

	# Check number of plots we need
	npl = int(
		np.ceil(
			np.sqrt( len(img_mat) * len(img_mat[0]) )))

	import pylab as plt
	from mpl_toolkits.axes_grid1 import ImageGrid
	fig = plt.figure(fignum, (8., 8.))
	fig.clf()
	fig.suptitle(pltit, fontsize=12)
	grid = ImageGrid(fig, 111, # similar to subplot(111)
					nrows_ncols = (npl, npl), # creates grid of axes
					axes_pad=0.1, # pad between axes in inch.
					)
	for xi, img_l in enumerate(img_mat):
		for xj, img in enumerate(img_l):
			grid[xi*npl+xj].imshow(img, **kwargs)

	plt.show()
	if (pause):
		raw_input()

def gauss(sz, spotsz, spotpos, amp, noiamp=0):
	"""
	Calculate 2D Gauss function.

	Gauss will be in a matrix of size **sz** with width **spotsz** at 
	position **spotpos** and amplitude **amp**. Poissonian noise will be 
	added with **noiamp** if > 0.

	@param [in] sz Size of output array
	@param [in] spotsz Size of Gaussian spot
	@param [in] spotpos Position of Gauss peak
	@param [in] amp Gauss amplitude
	@param [in] noiamp Poissonian noise amplitude
	@return ndarray of shape **sz** with the Guassian function
	"""

	# Coordinate grid
	r0 = (np.arange(sz[0]) - sz[0]/2.0).reshape(-1,1)
	r1 = (np.arange(sz[1]) - sz[1]/2.0).reshape(1,-1)

	# Make shifted Gaussian peak
	im = amp*np.exp(-((r0-spotpos[0])/spotsz)**2.0) * np.exp(-((r1-spotpos[1])/spotsz)**2.0)

	# Add noise if requested
	if (noiamp > 0):
		return im + np.random.poisson(noiamp, np.product(sz)).reshape(*sz)
	else:
		return im

def shift_img(im, shvec, method="pixel", zoomfac=8, apod=False, pad=False):
	"""
	Shift 2D array **im** by **shvec** using either pixel or Fourier method.

	# Pixel method

	Scale up image with scipy.ndimage.zoom() with a factor of 
	**zoomfac**, then shift by integer number of pixels, then zoom down 
	again to original size. The resolution of this shift is 1.0/**zoomfac**.

	# Fourier method
	
	Shift in Fourier space based on the Fourier transform shift theorem:

		f(x-dx,y-dy) <==> exp(-2pi i(u*dx+v*dy)) F(u,v)

	The Fourier shift code is taken from fftshiftcube written by Marshall 
	Perrin at 2001-07-27. Original api: FUNCTION 
	fftshiftcube,cube,dx,dy,null=null, but uses scipy.ndimage.fourier.
	fourier_shift for generating the complex shift vector.

	Documentation on the Fourier shift method might be available in [1], but 
	this algorithm was written independently.
	
	[1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008).

	@param [in] im 2D image to shift
	@param [in] shvec Vector to shift by in pixels
	@param [in] method Shifting method to use (Fourier or pixel)
	@param [in] zoomfac Zoom factor (for Pixel method)
	@param [in] apod Apodise image (for Fourier method)
	@param [in] pad Zero-pad data before FFT (for Fourier method)
	@return 2D image of identical shape as **im** shifted with **shvec**
	"""
	sz = im.shape

	if (method == "pixel"):
		# Blow up image and shift by zoomfac
		im_zm = scipy.ndimage.zoom(im, zoomfac)
		shvec_zm = np.round(np.r_[shvec]*zoomfac)

		# Calculate real shift vector
		realsh = np.round(np.r_[shvec]*zoomfac)/zoomfac

		# Shift scaled up image
		im_zm = np.roll(im_zm, int(shvec_zm[0]), axis=0)
		im_zm = np.roll(im_zm, int(shvec_zm[1]), axis=1)

		# Scale back to original size and return
		return scipy.ndimage.zoom(im_zm, 1./zoomfac)
	elif (method == "fourier"):
		if (pad):
			padfunc = _fft.embed_data
		else:
			padfunc = lambda x, direction=1: x

		apod_mask = 1
		if (apod):
			# Optional: apodise images.
			apod_mask = _fft.mk_apod_mask(im.shape, wsize=-0.1, apod_f='cos')
	
		# zero-pad data & FFT
		im_fft = np.fft.fft2(padfunc(im*apod_mask))

		# scipy.ndimage.fourier.fourier_shift multiplies an input array with 
		# a complex Fourier shifting vector and returns the multiplied data
		im_fft_sh = scipy.ndimage.fourier.fourier_shift(im_fft, shift=shvec)

		# IFFT, de-pad
		return padfunc(np.fft.ifft2(im_fft_sh).real, direction=-1)
	else:
		raise ValueError("<method> %s not valid" % (method))

def calc_subpixmax(data, offset=(0,0), dimension=2, index=False, error=False):
	"""
	Find extrema of **data** with subpixel accuracy.

	The subpixel maximum will be searched around the pixel with the maximum 
	intensity, i.e. numpy.argwhere(data == data.max())[0]. The coordinate 
	found will be in data-space.

	If **index** is True, return the CoG in element **index** coordinates, 
	i.,e. if the CoG is at element (5,1), return (5,1). Otherwise, return 
	the element **center** as coordinate, i.e. (5.5, 1.5) in the previous 
	example.

	For example, if we have this image:

      | 0| 0| 0| 0
	2-+--+--+--+--
      | 0| 0| 5| 0
	1-+--+--+--+--
      | 0| 0| 0| 0
	0-+--+--+--+--
      0  1  2  3  

    if index is True, CoG will be (2, 1), otherwise it will be (2.5, 1.5)

	For **dimension** == 2: use 9-point quadratic interpolation (QI formulae 
	by Yi & Molowny Horas (1992, Eq. (10)), also available in M.G. Löfdahl 
	(2010), table 2.).

	For **dimension** == 1: use 5-point 1D quadratic interpolation

	For **dimension** == 0: use the integer position of the maximum intensity pixel.

	If any interpolation failed, the next-lower dimension is tried.

	Warnings are shown if **error** is set.

	This routine is implemented in pure Python, formerly known quadInt2dPython

	@param [in] data Data to search for subpixel maximum
	@param [in] offset Add offset to output
	@param [in] dimension Interpolation dimensionality to use
	@param [in] index If True, return CoG in pixel **index** coordinate, otherwise return pixel **center** coordinate.
	@param [in] error Toggle error display
	"""
	if (not 0 <= dimension <= 2):
		raise ValueError("Interpolation <dimension> should be 0 <= d <= 2")

	off = 0.5
	if (index): off = 0

	# Initial guess for the interpolation
	s = np.argwhere(data == data.max())[0]

	# 0D: Maximum value pixel
	if (dimension == 0):
		return s - np.r_[offset] + off

	# If maximum position is at the edge, abort: we cannot calculate subpixel
	# maxima
	# np.B.: Maybe we should return the max pix position anyway?
	if ((s == 0).any() or (s+1 == data.shape).any()):
		return s - np.r_[offset] + off
#		raise ValueError("maximum value at edge of data, cannot calculate subpixel max")

	a2 = 0.5 * (data[ s[0]+1, s[1] ] - data[ s[0]-1, s[1] ])
	a3 = 0.5 * data[ s[0]+1, s[1] ] - data[ s[0], s[1] ] + \
		0.5 * data[ s[0]-1, s[1] ]
	a4 = 0.5 * (data[ s[0], s[1]+1 ] - data[ s[0], s[1]-1 ])
	a5 = 0.5 * data[ s[0], s[1]+1 ] - data[ s[0], s[1] ] + \
		0.5 * data[ s[0], s[1]-1 ]
	a6 = 0.25 * (data[ s[0]+1, s[1]+1 ] - data[ s[0]+1, s[1]-1 ] - \
		data[ s[0]-1, s[1]+1 ] + data[ s[0]-1, s[1]-1 ])

	# 2D Quadratic Interpolation
	if (dimension == 2):
		v = np.array([(2*a2*a5-a4*a6)/(a6*a6-4*a3*a5), \
			(2*a3*a4-a2*a6)/(a6*a6-4*a3*a5)])
		# Subpixel vector should be smaller than 1
		if ((np.abs(v) > 1).any()):
			if (error): print '!! 2D QI failed:', v
			dimension = 1
	# 1D Quadratic Interpolation
	# no elif here because we might need this from 2D fallback
	if (dimension == 1):
		v = np.array([a2/(2*a3), a4/(2*a5)])
		# Subpixel vector should be smaller than 1
		if ((np.abs(v) > 1).any()):
			if (error): print '!! 1D QI failed:', v
			dimension = 0
	# 0D: Maximum value pixel (keep here as fallback if 1D, 2D fail)
	# no elif here because we might need this from 1D fallback
	if (dimension == 0):
		return s - np.r_[offset] + off

	return v + s - np.r_[offset] + off

