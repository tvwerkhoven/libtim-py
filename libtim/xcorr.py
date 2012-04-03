#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@file xcorr.py
@brief Cross-correlation utils for measuring image shifts
@author Tim van Werkhoven (timvanwerkhoven@gmail.com)
@date 20120402

Created by Tim van Werkhoven (timvanwerkhoven@gmail.com) on 2012-04-02
Copyright (c) 2012 Tim van Werkhoven. All rights reserved.

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

import numpy as N

def crosscorr(imlst, shrange, refim=None):
	"""Cross-correlate images from <imlst> to measure image shifts in between

	@param [in] imlst list of images to cross correlate
	@param [in] shrange shift range to calculate, format ((-sh0, +sh0, dsh0), (-sh1, +sh1, dsh1))
	@param [in] refim Reference image to cross correlate others against. If None, will use cross correlate all pairs from imlst
	"""

	# Check if imlst is malformed
	try:
		test = imlst[0][0,0]*1
	except:
		raise ValueError("<imlst> should be a list of 2D arrays.")

	if (refim != None):
		if (refim.shape != imlst[0].shape):
			raise ValueError("<refim> should be same size as <imlst[0]>")

	# Convenience variables for shift range in two directions, 0 and 1
	sh00, sh01, dsh0 = shrange[0]
	sh10, sh11, dsh1 = shrange[1]

	# We need a crop window to allow for image shifting
	sm_crop = [slice(abs(sh00), -abs(sh01)), slice(abs(sh10), -abs(sh11))]

	print shrange, sm_crop

	# # Explicit full loop
	# for fi in imlst:
	# 	for fj in imlst:
	# 	# fj should be smaller because we move it over the larger window
	# 		for shi in xrange(shrange[0]):
	# 			for shj in xrange(shrange[1]):
	# 				fi[7+shi:7-shi, 7+shj:7-shj] * fj[sm_crop]

	# Calculate correlation for all files
	if (refim != None):
		xcorr_mat = [[ N.r_[ [[N.sum(refim[sh01+shi:sh00+shi, sh11+shj:sh10+shj] * refim[sm_crop])
			for shi in xrange(*shrange[0])]
				for shj in xrange(*shrange[1])] ]
					for fj in imlst]]
	else:
		xcorr_mat = [[ N.r_[ [[N.sum(fi[sh01+shi:sh00+shi, sh11+shj:sh10+shj] * fj[sm_crop])
			for shi in xrange(*shrange[0])]
				for shj in xrange(*shrange[1])] ]
					for fj in imlst]
						for fi in imlst]

	return xcorr_mat

def plot_img_mat(img_mat, fignum=0, pause=True, pltit="", titles=(), **kwargs):
	"""Plot grid of images

	@param [in] img_mat A matrix of 2D images
	@param [in] fignum pylab plot figure to use
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
	Shift <im> by <shvec> using either pixel or Fourier method.

	Pixel method: scale up image with scipy.ndimage.zoom() with a factor of <zoomfac>, then shift by integer number of pixels, then zoom down again to original size. The resolution of this shift is 1.0/<zoomfac>.

	Fourier method: shift in Fourier space.
	"""
	if (method == "pixel"):
		# Blow up image and shift by zoomfac
		im_zm = S.ndimage.zoom(im, zoomfac)
		sh_zm = N.round(N.r_[shvec]*zoomfac)

		# Calculate real shift vector
		realsh = N.round(N.r_[shvec]*zoomfac)/zoomfac

		# Shift scaled up image
		N.roll(im_zm, int(sh_zm[0]), axis=0)
		N.roll(im_zm, int(sh_zm[1]), axis=1)

		# Scale back to original size and return
		return S.ndimage.zoom(im_zm, 1./zoomfac)
	elif (method == "fourier"):
		raise NotImplemented("<method>=fourier not implemented")
		grid = N.indices(im.shape)

		# Make apodisation
		apod_mask = timlib.mk_apod_mask(im.shape, wsize=0.1, apod_f=wfunc)

		# FT image
		im_ft = N.fft.fft2(im)
	else:
		raise ValueError("<method> %s not valid" % (method))

def calc_subpixmax(data, offset=(0,0), dimension=2, error=False):
	"""
	Find the extrema of 'data' using a two-dimensional 9-point quadratic
	interpolation (QI formulae by Yi & Molowny Horas (1992, Eq. (10)), also
	available in M.G. Löfdahl (2010), table 2.). The subpixel maximum will be
	examined around the coordinate of the pixel with the maximum intensity.

	'offset' must be set to the shift-range that the correlation map in 'data'
	corresponds to to find the pixel corresponding to a shift of 0.

	'dimension' indicates the dimension of the quadratic interpolation, this can be either 2, 1 or 0 for no interpolation.

	If 2D quadratic interpolation fails (i.e. shift > 1), a 1D QI is done. If
	this fails (i.e. shift > 1), the integer coordinates of the pixel with the
	maximum intensity is returned.

	Warnings are shown if 'error' is set.

	This routine is implemented in pure Python, formerly known quadInt2dPython
	"""
	if (not 0 <= dimension <= 2):
		raise ValueError("Interpolation <dimension> should be 0 <= d <= 2")

	# Initial guess for the interpolation
	s = N.argwhere(data == data.max())[0]

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
	elif (dimension == 1):
		v = N.array([a2/(2*a3), a4/(2*a5)])
		# Subpixel vector should be smaller than 1
		if ((N.abs(v) > 1).any()):
			if (error): print '!! 1D QI failed:', v
			dimension = 0
	# Maximum value pixel
	elif (dimension == 0):
		v = N.array([0 , 0])

	return v + s - N.r_[offset]

### Testing begins here

import unittest

class TestXcorr(unittest.TestCase):
	def setUp(self):
		"""Generate test data"""

		import scipy as S
		import scipy.ndimage			# For fast ndimage processing

		### Test parameters ###
		# Test image size
		sz_l = [(37, 43), (61, 31)]
# 		sz_l = [(37, 43), (257, 509)]
		# Gaussian spot size
		spotsz = 5.
		# Test shift vectors
		self.shtest = [(0,0), (5.5, 4.3), (0.9, 0.8), (3.2, 11.1)]
		# Poisson noise factor and image intensity factor
		nfac = 75
		imfac = 255

		### Generate random test images ###
		# Use random patterns as input. Scale up a factor <zoomfac>, then shift images, then scale back

# 		zoomfac=8
# 		self.testimg_l2 = []
# 		self.shtest2 = []
# 		for sz in sz_l:
# 			sz2 = (N.r_[sz]/2.0).reshape(2,1,1)
# 			randim = N.random.random(sz)*255
# 			testimg_ll = []
# 			# Blow up image by zoomfac
# 			randim_zm = S.ndimage.zoom(randim, zoomfac)
#
# 			# Shift images in zoomed space by integer shifts
# 			for sh in self.shtest:
# 				# We cannot shift arbitrary vectors, exact shifts depends on
# 				# zoom factor
# 				thissh_zm = N.round(N.r_[sh]*zoomfac)
# 				# Store real shifts
# 				self.shtest2.append(tuple(thissh_zm/zoomfac))
#
# 				# Shift image (wrap around)
# 				thisim = randim_zm.copy()
# 				N.roll(thisim, int(thissh_zm[0]), axis=0)
# 				N.roll(thisim, int(thissh_zm[1]), axis=1)
#
# 				# Zoom down (again by zoomfac), store
# 				thisim2 = S.ndimage.zoom(thisim, 1./zoomfac)
# 				testimg_ll.append(thisim2)
# 			# Show images
# # 			plot_img_mat([testimg_ll], pause=True, extent=(-sz[1]/2, sz[1]/2, sz[0]/2, -sz[0]/2))
# 			self.testimg_l2.append(testimg_ll)

		### Generate test images ###
		# Use Gaussian peaks to test image shifts

		# Loop over all test sizes and shift vectors to generate images
		self.testimg_l = []
		for sz in sz_l:
			# Half the size to generate an image
			sz2 = (N.r_[sz]/2.0).reshape(2,1,1)
			grid = (N.indices(sz, dtype=N.float) - sz2)
			testimg_ll = []
			for sh in self.shtest:
				# Generate Poisson noise
				noi = N.random.poisson(nfac, N.product(sz))
				# Make shifted Gaussian peak, add noise
				im = imfac*N.exp(-((grid[0]-sh[0])/spotsz)**2.0) * N.exp(-((grid[1]-sh[1])/spotsz)**2.0) + noi.reshape(*sz)
				# Store in list
				testimg_ll.append(im)
			# Show images
# 			plot_img_mat([testimg_ll], pause=True, extent=(-sz[1]/2, sz[1]/2, sz[0]/2, -sz[0]/2))
			self.testimg_l.append(testimg_ll)


	# Shallow data tests
	def test0a_inspect(self):
		"""Dummy test, inspect data"""
		# Loop over different sizes
		for idx, testimg_l0 in enumerate(self.testimg_l):
			# Plot images for this size
			sz = testimg_l0[0].shape
			tit = "Input data for sz="+str(sz)+"\nshifts:"+str(self.shtest)
			plot_img_mat([testimg_l0], fignum=idx, pause=True, pltit=tit, extent=(-sz[1]/2, sz[1]/2, sz[0]/2, -sz[0]/2))


	def test1a_xcorr(self):
		"""Test inter-image cross-correlation"""
		# Loop over different sizes
		for idx,testimg_l0 in enumerate(self.testimg_l):
			# Test this shiftrange
			shr = ((-5,3,1), (-5, 7, 1))

			sz = testimg_l0[0].shape

			# Output is a xcorr for all image pairs
			outarr = crosscorr(testimg_l0, shr, refim=None)

			# Plot correlation maps
			tit = "Input data for sz="+str(sz)+"\nshifts:"+str(self.shtest)
			plot_img_mat(outarr, fignum=idx+10, pause=True, pltit=tit, extent=(shr[0][0], shr[0][1], shr[1][0], shr[1][1]))


if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())
