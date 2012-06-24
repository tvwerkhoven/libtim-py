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

#=============================================================================
# Import libraries here
#=============================================================================

import numpy as N
import unittest
import fft as _fft

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def crosscorr(imlst, shrange, dsh=(1,1), refim=None):
	"""
	Cross-correlate images in **imlst**.

	**shrange** determines the range of shifts to cross-correlate for, and **dsh** indicates the step-size to use, i.e. dsh = (2,2) means every second shift distance is used.

	If **refim** is given, compare all images from **imlist** with this image, otherwise cross-correlate all pairs of images in **imlist**.

	The returned data is a list of lists with correlation maps. The correlation maps themselves have shape (2*shrange+1)/dsh. If refim is given, the list of lists is Nx1 big, otherwise it is NxN big.

	Correlation maps are only square if shrange[0] == shrange[1] and dsh[0] == dsh[1].

	@param [in] imlst list of images to cross correlate
	@param [in] shrange shift range to calculate, format (sh0, sh1)
	@param [in] dsh shift range cadence
	@param [in] refim Reference image to cross correlate others against. If None, will use cross correlate all pairs from imlst
	@return NxN (refim==None) or Nx1 (refim!=None) list of cross-correlation maps. Each map is (2*shrange+1)/dsh big)
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
	if (sh0 < 1 or sh1 < 1):
		raise ValueError("<shrange> should be larger than 0")

	# We need a crop window to allow for image shifting
	sm_crop = [slice(sh0, -sh0), slice(sh1, -sh1)]

	# Calculate correlation for all files with <refim> (output Nx1)
	if (refim != None):
		xcorr_mat = [[
			N.r_[ [[
			N.sum(refim[sh0+shi:imsz0-sh0+shi, sh1+shj:imsz1-sh1+shj] * fj[sm_crop])
			for shj in xrange(-sh1, sh1+1, dsh1)]
				for shi in xrange(-sh0, sh0+1, dsh0)]
				] # // N.r_
					for fj in imlst]]
	# Calculate correlation for all files with each other (output NxN)
	else:
		xcorr_mat = [[
			N.r_[ [[
			N.sum(fi[sh0+shi:imsz0-sh0+shi, sh1+shj:imsz1-sh1+shj] * fj[sm_crop])
			for shj in xrange(-sh1, sh1+1, dsh1)]
				for shi in xrange(-sh0, sh0+1, dsh0)]
				] # // N.r_
					for fj in imlst]
						for fi in imlst]

	return xcorr_mat

def crosscorr1(img, refim, shrange, dsh=(1,1)):
	"""
	Calculate cross-correlation for only 1 image.
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
	xcorr = N.r_[ [[
		N.sum(refim[sh0+shi:imsz0-sh0+shi, sh1+shj:imsz1-sh1+shj] * img[sm_crop])
		for shj in xrange(-sh1, sh1+1, dsh1)]
			for shi in xrange(-sh0, sh0+1, dsh0)]
				] # // N.r_

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
		N.ceil(
			N.sqrt( len(img_mat) * len(img_mat[0]) )))

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

def shift_img(im, shvec, method="pixel", zoomfac=8):
	"""
	Shift 2D array **im** by **shvec** using either pixel or Fourier method.

	Pixel method: scale up image with scipy.ndimage.zoom() with a factor of **zoomfac**, then shift by integer number of pixels, then zoom down again to original size. The resolution of this shift is 1.0/**zoomfac**.

	Fourier method: shift in Fourier space based on the Fourier transform
	shift theorem:
		f(x-dx,y-dy) <==> exp(-2pi i(u*dx+v*dy)) F(u,v)

	The Fourier shift code is taken from fftshiftcube written by Marshall Perrin at 2001-07-27. Original api: FUNCTION fftshiftcube,cube,dx,dy,null=null

	@param [in] im 2D image to shift
	@param [in] shvec Vector to shift by
	@param [in] method Shifting method to use
	@param [in] zoomfac Zoom factor for pixel shifting method
	@return 2D image of identical shape as **im** shifted with **shvec**
	"""
	sz = im.shape

	if (method == "pixel"):
		import scipy as S
		import scipy.ndimage
		# Blow up image and shift by zoomfac
		im_zm = S.ndimage.zoom(im, zoomfac)
		shvec_zm = N.round(N.r_[shvec]*zoomfac)

		# Calculate real shift vector
		realsh = N.round(N.r_[shvec]*zoomfac)/zoomfac

		# Shift scaled up image
		im_zm = N.roll(im_zm, int(shvec_zm[0]), axis=0)
		im_zm = N.roll(im_zm, int(shvec_zm[1]), axis=1)

		# Scale back to original size and return
		return S.ndimage.zoom(im_zm, 1./zoomfac)
	elif (method == "fourier"):
		# Linear increasing array, subtract 50%, roll 50% -> sawtooth
		u0 = N.roll(N.arange(sz[0])*1. - sz[0]/2, sz[0]/2).reshape(-1,1)
		u1 = N.roll(N.arange(sz[1])*1. - sz[0]/2, sz[1]/2).reshape(1,-1)

		# Convert to frequency, mult by shift
		u0 = shvec[0] * u0/sz[0]
		u1 = shvec[1] * u1/sz[1]

		# Make apodisation
		apod_mask = _fft.mk_apod_mask(im.shape, wsize=-0.1, apod_f='cos')

		# Mask images. Use the median not the mean because it's better at
		# grabbing the sky rather than the star
		offs = N.median(im)
		im2 = (im-offs) * apod_mask

		# FT image
		im_ft = N.fft.fft2(im2)

		# Make shift mask
		sh_mask = N.exp(-2*N.pi*1j*(u0+u1))

		# Shift and return
		return N.fft.ifft2(sh_mask*im_ft).real + offs
	else:
		raise ValueError("<method> %s not valid" % (method))

def calc_subpixmax(data, offset=(0,0), dimension=2, error=False):
	"""
	Find extrema of **data** with subpixel accuracy.

	The subpixel maximum will be searched around the pixel with the maximum intensity, i.e. numpy.argwhere(data == data.max())[0]. The coordinate found will be in data-space.

	For **dimension** == 2: use 9-point quadratic interpolation (QI formulae by Yi & Molowny Horas (1992, Eq. (10)), also available in M.G. Löfdahl (2010), table 2.).

	For **dimension** == 1: use 5-point 1D quadratic interpolation

	For **dimension** == 0: use the integer position of the maximum intensity pixel.

	If any interpolation failed, the next-lower dimension is tried.

	Warnings are shown if **error** is set.

	This routine is implemented in pure Python, formerly known quadInt2dPython

	@param [in] data Data to search for subpixel maximum
	@param [in] offset Add offset to output
	@param [in] dimension Interpolation dimensionality to use
	@param [in] error Toggle error display
	"""
	if (not 0 <= dimension <= 2):
		raise ValueError("Interpolation <dimension> should be 0 <= d <= 2")

	# Initial guess for the interpolation
	s = N.argwhere(data == data.max())[0]

	# 0D: Maximum value pixel
	if (dimension == 0):
		return s - N.r_[offset]

	# If maximum position is at the edge, abort: we cannot calculate subpixel
	# maxima
	# N.B.: Maybe we should return the max pix position anyway?
	if ((s == 0).any() or (s+1 == data.shape).any()):
		return s - N.r_[offset]
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
		v = N.array([(2*a2*a5-a4*a6)/(a6*a6-4*a3*a5), \
			(2*a3*a4-a2*a6)/(a6*a6-4*a3*a5)])
		# Subpixel vector should be smaller than 1
		if ((N.abs(v) > 1).any()):
			if (error): print '!! 2D QI failed:', v
			dimension = 1
	# 1D Quadratic Interpolation
	# no elif here because we might need this from 2D fallback
	if (dimension == 1):
		v = N.array([a2/(2*a3), a4/(2*a5)])
		# Subpixel vector should be smaller than 1
		if ((N.abs(v) > 1).any()):
			if (error): print '!! 1D QI failed:', v
			dimension = 0
	# 0D: Maximum value pixel (keep here as fallback if 1D, 2D fail)
	# no elif here because we might need this from 1D fallback
	if (dimension == 0):
		return s - N.r_[offset]

	return v + s - N.r_[offset]

#=============================================================================
# Unittesting
#=============================================================================

def _gauss_slow(sz, spotsz, spotpos, amp, noiamp):
	"""
	@deprecated please use _gauss() instead which is ~3--20x faster.
	"""

	# Coordinate grid
	sz2 = (N.r_[sz]/2.0).reshape(2,1,1)
	grid = (N.indices(sz, dtype=N.float) - sz2)

	# Make shifted Gaussian peak
	im = amp*N.exp(-((grid[0]-spotpos[0])/spotsz)**2.0) * N.exp(-((grid[1]-spotpos[1])/spotsz)**2.0)

	# Add noise if requested
	if (noiamp > 0):
		return im + N.random.poisson(noiamp, N.product(sz)).reshape(*sz)
	else:
		return im

def _gauss(sz, spotsz, spotpos, amp, noiamp=0):
	"""
	Calculate 2D Gauss function.

	Gauss will be in a matrix of size **sz** with width **spotsz** at position **spotpos** and amplitude **amp**. Poissonian noise will be added with **noiamp** if > 0.

	@param [in] sz Size of output array
	@param [in] spotsz Size of Gaussian spot
	@param [in] spotpos Position of Gauss peak
	@param [in] amp Gauss amplitude
	@param [in] noiamp Poissonian noise amplitude
	@return ndarray of shape **sz** with the Guassian function
	"""

	# Coordinate grid
	r0 = (N.arange(sz[0]) - sz[0]/2.0).reshape(-1,1)
	r1 = (N.arange(sz[1]) - sz[1]/2.0).reshape(1,-1)

	# Make shifted Gaussian peak
	im = amp*N.exp(-((r0-spotpos[0])/spotsz)**2.0) * N.exp(-((r1-spotpos[1])/spotsz)**2.0)

	# Add noise if requested
	if (noiamp > 0):
		return im + N.random.poisson(noiamp, N.product(sz)).reshape(*sz)
	else:
		return im

class TestShift(unittest.TestCase):
	def setUp(self):
		# Make init gauss
		self.sz = (257, 509)
		self.sz = (31, 29)
		self.spotsz = 2.
		self.pos_l = [(1,1), (2*self.spotsz, -2*self.spotsz), (5,8)]
		self.amp = 255

		self.refim = _gauss(self.sz, self.spotsz, (0,0), self.amp, 0)

	# shift_img(im, shvec, method="pixel", zoomfac=8)
	def test1a_identity(self):
		"""Test if move function does not change a (0,0) shift"""
		shift = (0,0)
		im_sh1 = shift_img(self.refim, shift, method="pixel", zoomfac=8)
		im_sh2 = shift_img(self.refim, shift, method="fourier")

		self.assertLess(N.mean(im_sh1-self.refim), 1e-6)
		self.assertLess(N.mean(im_sh2-self.refim), 1e-6)


	def test1b_diff(self):
		"""Test if a non-zero shift changes the image"""
		shift = (2*self.spotsz, 2*self.spotsz)
		im_sh1 = shift_img(self.refim, shift, method="pixel", zoomfac=8)
		im_sh2 = shift_img(self.refim, shift, method="fourier")

		# Total summed diff should be zero because the shape doesn't change
		# Pixel shifting is not very accurate, so the residual is bigger
		self.assertLess(N.mean(im_sh1-self.refim), self.amp/100.)
		self.assertLess(N.mean(im_sh2-self.refim), 1e-6)

		# The absolute difference should be very nonzero
		self.assertGreater(N.sum(N.abs(im_sh1-self.refim)), self.amp)
		self.assertGreater(N.sum(N.abs(im_sh2-self.refim)), self.amp)

	def test2a_plot(self):
		"""Plot shifted images"""

		plrn = (-self.sz[0]/2, self.sz[0]/2, -self.sz[1]/2, self.sz[1]/2)
		plt.figure(0)
		plt.clf()
		plt.title("Reference image")
		plt.imshow(self.refim, extent=plrn)
		for sh in self.pos_l:
			im_sh1 = shift_img(self.refim, sh, method="pixel", zoomfac=8)
			im_sh2 = shift_img(self.refim, sh, method="fourier")
			plt.figure(1)
			plt.clf()
			plt.title("Pixel-shifted with: " + str(sh))
			plt.imshow(im_sh1, extent=plrn)
			plt.figure(2)
			plt.clf()
			plt.title("Fourier-shifted with: " + str(sh))
			plt.imshow(im_sh2, extent=plrn)
			raw_input()

class TestGaussFuncs(unittest.TestCase):
	def setUp(self):
		# Gauss function settings
		self.sz_l = [(37, 43), (257, 509)]
		self.spotsz_l = [1.,8.]
		self.pos_l = [(1,1), (15,13), (20,1)]
		self.amp = 255
		self.noi = 0
		# Timing parameters
		self.niter = 100

	# api: _gauss[2](sz, spotsz, spotpos, amp, noiamp)

	def test1a_equal(self):
		"""Test if two functions give identical results"""
		for sz in self.sz_l:
			for spsz in self.spotsz_l:
				for pos in self.pos_l:
					g1 = _gauss_slow(sz, spsz, pos, self.amp, self.noi)
					g2 = _gauss(sz, spsz, pos, self.amp, self.noi)
					##! @todo Proper way to assert two ndarrays identicity?
					#print sz, spsz, pos, N.mean(g1-g2), 0.0
					self.assertAlmostEqual(N.mean(g1-g2), 0.0)
					self.assertTrue(N.allclose(g1, g2))

	def test2a_timing(self):
		"""Test timing for two functions"""
		print "test2a_timing(): timings in msec/iter"
		for sz in self.sz_l:
			for spsz in self.spotsz_l:
				for pos in self.pos_l:
					setup_str = """
from __main__ import _gauss, _gauss_slow
import numpy as N
sz = (%d,%d)
spsz = %g
pos = (%d,%d)
amp = %g
noi = %g
					""" % (sz + (spsz,) + pos + (self.amp, self.noi))

					t1 = Timer("""
g=_gauss_slow(sz, spsz, pos, amp, noi)
					""", setup_str)
					t2 = Timer("""
a=_gauss(sz, spsz, pos, amp, noi)
					""", setup_str)
					t_g1 = 1000*min(t1.repeat(3, self.niter))/self.niter
					t_g2 = 1000*min(t2.repeat(3, self.niter))/self.niter
					print "test2a_timing(): sz:", sz, "g1: %.3g, g2: %.3g, speedup: %.3g" % (t_g1, t_g2, t_g1/t_g2)


class TestSubpixmax(unittest.TestCase):
	def setUp(self):
		self.sz_l = [(37, 43), (61, 31)]
		self.pos_l = [(1,5), (15,13), (20,1), (35, 29)]

	# api: calc_subpixmax(data, offset=(0,0), dimension=2, error=False):

	def test1a_simple(self):
		"""Simple test of hot pixels in zero matrix"""
		for sz in self.sz_l:
			for pos in self.pos_l:
				simple = N.zeros(sz)
				simple[pos] = 1
				for dim in range(3):
					p = calc_subpixmax(simple, dimension=dim)
					self.assertEqual(tuple(p), pos)

	def test1b_offset(self):
		"""Simple test of hot pixels in zero matrix, using offset"""
		for sz in self.sz_l:
			offs = N.r_[sz]/2.
			for pos in self.pos_l:
				simple = N.zeros(sz)
				simple[pos] = 1
				for dim in range(3):
					p = calc_subpixmax(simple, offset=offs, dimension=dim)
					self.assertEqual(tuple(p), tuple(N.r_[pos]-offs))

	def test3a_random_data(self):
		"""Give random input data"""
		for it in range(10):
			rnd = N.random.random(self.sz_l[0])
			# Set edge to zero to prevent erros
			rnd[0] = rnd[-1] = 0
			rnd[:,0] = rnd[:,-1] = 0
			# Don't print errors because output is garbage anyway
			for dim in range(3):
				calc_subpixmax(rnd, offset=(0,0), dimension=dim, error=False)

	def test3b_edge_error(self):
		"""Maximum at edge should give max intensity pixel pos"""
		map = N.random.random(self.sz_l[0])*0.7
		# Set maximum at the edge
		map[0,0] = 1
		# 0D should not raise:
		vec0 = calc_subpixmax(map, offset=(0,0), dimension=0, error=True)
		# 1D and 2D should raise:
# 		with self.assertRaises(ValueError):
# 			calc_subpixmax(map, offset=(0,0), dimension=1, error=True)
# 		with self.assertRaises(ValueError):
# 			calc_subpixmax(map, offset=(0,0), dimension=2, error=True)
		vec1 = calc_subpixmax(map, offset=(0,0), dimension=1, error=True)
		vec2 = calc_subpixmax(map, offset=(0,0), dimension=2, error=True)
		self.assertEqual(tuple(vec0), (0,0))
		self.assertEqual(tuple(vec0), tuple(vec1))
		self.assertEqual(tuple(vec0), tuple(vec2))

class BaseXcorr(unittest.TestCase):
	def setUp(self):
		"""Generate test data"""
		### Test parameters ###
		# Test image size
		sz_l = [(37, 43), (61, 31)]
		sz_l = [(37, 43), (257, 509)]
		# Test shift vectors
		# NB: If these are too large compared to shrng and sz_l, we cannot
		# find it back with xcorr measurements
#		self.shtest = [(0,0), (5.5, 4.3), (0.9, 0.8), (3.2, 11.1)]
		self.shtest = [(0,0), (5,2), (1.5981882, 2.312351), (0.9, 0.8)]
		# Poisson noise factor and image intensity factor
		self.nfac = 5
		self.imfac = 255
		# Gaussian spot size
		spotsz = 5.
		# Shift range to measure
#		self.shrng = (18, 16)
		self.shrng = (6, 6)

		### Generate Gaussian test images ###
		# Loop over all test sizes and shift vectors to generate images, store in self.testimg_l
		self.testimg_l = [[_gauss(sz, spotsz, sh, self.imfac, self.nfac)
			for sh in self.shtest] for sz in sz_l]

class TestXcorr(BaseXcorr):
	def test1a_test_func(self):
		"""Test cross-correlation function calls"""
		# Loop over different sizes
		for idx, testimg_l0 in enumerate(self.testimg_l):
			refim = testimg_l0[0]
			for shr in [(1,1), (5,5), (3,8), (1,9)]:
				for dsh in [(1,1), (2,2)]:
					corr_map = crosscorr(testimg_l0, shr, dsh, refim)
					# Correlation with refim should give 1xN correlation list
					self.assertEqual(len(corr_map), 1)
					self.assertEqual(len(corr_map[0]), len(testimg_l0))

					# Individual corr maps should be ceil(2*shr+1 / dsh)
					self.assertEqual(
						corr_map[0][0].shape,
						tuple( (N.ceil((2.0*N.r_[shr]+1)/N.r_[dsh])) )
						) # //assertEqual

					corr_map = crosscorr(testimg_l0, shr, dsh)
					# Without refim should give a NxN correlation list
					self.assertEqual(len(corr_map), len(corr_map[0]))


	def test2a_xcorr(self):
		"""Test inter-image cross-correlation"""
		# Loop over different sizes
		for idx,testimg_l0 in enumerate(self.testimg_l):
			# Test this shiftrange
			shr = self.shrng
			sz = testimg_l0[0].shape
			sz2 = N.r_[sz]/2.

			# Output is a cross-corr for all image pairs
			outarr = crosscorr(testimg_l0, shr, refim=None)

			# Loop over shift vectors and correlation maps
			for shi, outarr_l in zip(self.shtest, outarr):
				for shj, corr in zip(self.shtest, outarr_l):
					vec = calc_subpixmax(corr, offset=N.r_[corr.shape]/2)

					# In some cases, the shift is too large to measure. This
					# happens when the shift is larger than the shr or it is
					# outside the cross-correlated area
					if (shi[0]+shj[0]+1 > sz2[0]-shr[0] or
						shi[1]+shj[1]+1 > sz2[0]-shr[1] or
						shi[0]+shj[0]+1 > shr[0] or
						shi[1]+shj[1]+1 > shr[1]):
						# This shift might be too large to measure wrt the shift range and image size, don't test
						pass
					elif (self.nfac <= 5):
						self.assertTrue(shi[0]-shj[0]-vec[0] < 0.05)
						self.assertTrue(shi[1]-shj[1]-vec[1] < 0.05)

class PlotXcorr(BaseXcorr):
	# Shallow data tests
	def test0a_inspect(self):
		"""Dummy test, inspect data"""
		# Loop over different sizes
		for idx, testimg_l0 in enumerate(self.testimg_l):
			shr = self.shrng
			# Plot images for this size
			sz = testimg_l0[0].shape
			tit = "Input data for sz="+str(sz) + "shr="+str(shr) + "\nshifts:"+str(self.shtest)
			plot_img_mat([testimg_l0], fignum=idx, pause=True, pltit=tit, extent=(-sz[1]/2, sz[1]/2, sz[0]/2, -sz[0]/2))

	def test1a_xcorr(self):
		"""Plot inter-image cross-correlation"""
		# Loop over different sizes
		import pylab as plt
		for idx,testimg_l0 in enumerate(self.testimg_l):
			# Test this shiftrange
			shr = self.shrng
			sz = testimg_l0[0].shape

			# Output is a xcorr for all image pairs
			outarr = crosscorr(testimg_l0, shr, refim=testimg_l0[0])
			vec = calc_subpixmax(outarr, offset=N.r_[outarr.shape]/2)

			# Plot correlation maps
			tit = "Xcorr maps for sz="+str(sz) + "shr="+str(shr) + "\nshifts:"+str(self.shtest)
			plot_img_mat(outarr, fignum=idx+10, pause=True, pltit=tit, extent=(-shr[0], shr[0], -shr[1], shr[1]))


	def test1b_xcorr_sh(self):
		"""Plot inter-image cross-correlation shifts"""
		# Loop over different sizes
		import pylab as plt
		for idx,testimg_l0 in enumerate(self.testimg_l):
			# Test this shiftrange
			shr = self.shrng
			sz = testimg_l0[0].shape

			# Output is a xcorr for all image pairs
			outarr = crosscorr(testimg_l0, shr, refim=None)

			plt.figure(0)
			plt.clf()
			tit = "Xcorr shifts for sz="+str(sz) + "shr="+str(shr) + "\nshifts:"+str(self.shtest)
			plt.title(tit)
			for shi, outarr_l in zip(self.shtest, outarr):
				for shj, corr in zip(self.shtest, outarr_l):
# 					print shi, shj
# 					print N.r_[shi]-N.r_[shj]
					vec = calc_subpixmax(corr, offset=N.r_[corr.shape]/2)
					plt.plot([shi[0]-shj[0], vec[0]], [shi[1]-shj[1], vec[1]], '+-')
			plt.figure(1)
			plt.clf()
			tit = "Xcorr shifts resid for sz="+str(sz) + "shr="+str(shr) + "\nshifts:"+str(self.shtest)
			plt.title(tit)
			for shi, outarr_l in zip(self.shtest, outarr):
				for shj, corr in zip(self.shtest, outarr_l):
# 					print shi, shj
# 					print N.r_[shi]-N.r_[shj]
					vec = calc_subpixmax(corr, offset=N.r_[corr.shape]/2)
					plt.plot([shi[0]-shj[0]-vec[0]], [shi[1]-shj[1]-vec[1]], '*')
			raw_input()

if __name__ == "__main__":
	import sys
	from timeit import Timer
	import pylab as plt
	sys.exit(unittest.main())
