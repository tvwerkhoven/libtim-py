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

import unittest
import fft as _fft
from xcorr import *

SHOWPLOTS=False

#==========================================================================
# Unittesting
#==========================================================================

def _gauss_slow(sz, spotsz, spotpos, amp, noiamp):
	"""
	@deprecated please use gauss() instead which is ~3--20x faster.
	"""

	# Coordinate grid
	sz2 = (np.r_[sz]/2.0).reshape(2,1,1)
	grid = (np.indices(sz, dtype=np.float) - sz2)

	# Make shifted Gaussian peak
	im = amp*np.exp(-((grid[0]-spotpos[0])/spotsz)**2.0) * np.exp(-((grid[1]-spotpos[1])/spotsz)**2.0)

	# Add noise if requested
	if (noiamp > 0):
		return im + np.random.poisson(noiamp, np.product(sz)).reshape(*sz)
	else:
		return im

class TestShift(unittest.TestCase):
	def setUp(self):
		# Make init gauss
		self.sz = (257, 509)
		self.sz = (31, 29)
		self.spotsz = 2.
		self.pos_l = [(1,1), (2**0.5*self.spotsz, 2**0.5), (2*self.spotsz, -2*self.spotsz), (5,8)]
		self.amp = 255

		self.refim = gauss(self.sz, self.spotsz, (0,0), self.amp, 0)

	# shift_img(im, shvec, method="pixel", zoomfac=8)
	def test1a_identity(self):
		"""Test if move function does not change a (0,0) shift"""
		shift = (0,0)
		im_sh1 = shift_img(self.refim, shift, method="pixel", zoomfac=8)
		im_sh2 = shift_img(self.refim, shift, method="fourier")

		self.assertLess(np.mean(im_sh1-self.refim), 1e-6)
		self.assertLess(np.mean(im_sh2-self.refim), 1e-6)


	def test1b_diff(self):
		"""Test if a non-zero shift changes the image"""
		shift = (2*self.spotsz, 2*self.spotsz)
		im_sh1 = shift_img(self.refim, shift, method="pixel", zoomfac=8)
		im_sh2 = shift_img(self.refim, shift, method="fourier")

		# Total summed diff should be zero because the shape doesn't change
		# Pixel shifting is not very accurate, so the residual is bigger
		self.assertLess(np.mean(im_sh1-self.refim), self.amp/100.)
		self.assertLess(np.mean(im_sh2-self.refim), 1e-6)

		# The absolute difference should be very nonzero
		self.assertGreater(np.sum(np.abs(im_sh1-self.refim)), self.amp)
		self.assertGreater(np.sum(np.abs(im_sh2-self.refim)), self.amp)

	def test2a_plot(self):
		"""Plot shifted images"""

		if (not SHOWPLOTS):
			return
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
			plt.title("Pixel-shifted by (%.3g, %.3g)" % sh)
			plt.imshow(im_sh1, extent=plrn)
			plt.figure(2)
			plt.clf()
			plt.title("Fourier-shifted by (%.3g, %.3g)" % sh)
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
					g2 = gauss(sz, spsz, pos, self.amp, self.noi)
					##! @todo Proper way to assert two ndarrays identicity?
					#print sz, spsz, pos, np.mean(g1-g2), 0.0
					self.assertAlmostEqual(np.mean(g1-g2), 0.0)
					self.assertTrue(np.allclose(g1, g2))

	def test2a_timing(self):
		"""Test timing for two functions"""
		print "test2a_timing(): timings in msec/iter"
		for sz in self.sz_l[:1]:
			for spsz in self.spotsz_l:
				for pos in self.pos_l:
					setup_str = """
from __main__ import gauss, _gauss_slow
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
a=gauss(sz, spsz, pos, amp, noi)
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
				simple = np.zeros(sz)
				simple[pos] = 1
				for dim in range(3):
					p = calc_subpixmax(simple, dimension=dim)
					self.assertEqual(tuple(p), pos)

	def test1b_offset(self):
		"""Simple test of hot pixels in zero matrix, using offset"""
		for sz in self.sz_l:
			offs = np.r_[sz]/2.
			for pos in self.pos_l:
				simple = np.zeros(sz)
				simple[pos] = 1
				for dim in range(3):
					p = calc_subpixmax(simple, offset=offs, dimension=dim)
					self.assertEqual(tuple(p), tuple(np.r_[pos]-offs))

	def test3a_random_data(self):
		"""Give random input data"""
		for it in range(10):
			rnd = np.random.random(self.sz_l[0])
			# Set edge to zero to prevent erros
			rnd[0] = rnd[-1] = 0
			rnd[:,0] = rnd[:,-1] = 0
			# Don't print errors because output is garbage anyway
			for dim in range(3):
				calc_subpixmax(rnd, offset=(0,0), dimension=dim, error=False)

	def test3b_edge_error(self):
		"""Maximum at edge should give max intensity pixel pos"""
		map = np.random.random(self.sz_l[0])*0.7
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
		self.testimg_l = [[gauss(sz, spotsz, sh, self.imfac, self.nfac)
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
						tuple( (np.ceil((2.0*np.r_[shr]+1)/np.r_[dsh])) )
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
			sz2 = np.r_[sz]/2.

			# Output is a cross-corr for all image pairs
			outarr = crosscorr(testimg_l0, shr, refim=None)

			# Loop over shift vectors and correlation maps
			for shi, outarr_l in zip(self.shtest, outarr):
				for shj, corr in zip(self.shtest, outarr_l):
					corr = np.r_[corr]
					vec = calc_subpixmax(corr, offset=np.r_[corr.shape]/2)

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
			if (SHOWPLOTS):
				plot_img_mat([testimg_l0], fignum=idx, pause=True, pltit=tit, extent=(-sz[1]/2, sz[1]/2, sz[0]/2, -sz[0]/2))

	def test1a_xcorr(self):
		"""Plot inter-image cross-correlation"""
		# Loop over different sizes
		if (not SHOWPLOTS):
			return
		import pylab as plt
		for idx,testimg_l0 in enumerate(self.testimg_l):
			# Test this shiftrange
			shr = self.shrng
			sz = testimg_l0[0].shape

			# Output is a xcorr for all image pairs
			outarr = crosscorr(testimg_l0, shr, refim=testimg_l0[0])
			outarr = np.r_[outarr]
			vec = calc_subpixmax(outarr, offset=np.r_[outarr.shape]/2)

			# Plot correlation maps
			tit = "Xcorr maps for sz="+str(sz) + "shr="+str(shr) + "\nshifts:"+str(self.shtest)
			plot_img_mat(outarr, fignum=idx+10, pause=True, pltit=tit, extent=(-shr[0], shr[0], -shr[1], shr[1]))


	def test1b_xcorr_sh(self):
		"""Plot inter-image cross-correlation shifts"""
		# Loop over different sizes
		if (not SHOWPLOTS):
			return
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
# 					print np.r_[shi]-np.r_[shj]r_
					corr = np.r_[corr]
					vec = calc_subpixmax(corr, offset=np.r_[corr.shape]/2)
					plt.plot([shi[0]-shj[0], vec[0]], [shi[1]-shj[1], vec[1]], '+-')
			plt.figure(1)
			plt.clf()
			tit = "Xcorr shifts resid for sz="+str(sz) + "shr="+str(shr) + "\nshifts:"+str(self.shtest)
			plt.title(tit)
			for shi, outarr_l in zip(self.shtest, outarr):
				for shj, corr in zip(self.shtest, outarr_l):
# 					print shi, shj
# 					print np.r_[shi]-np.r_[shj]
					corr = np.r_[corr]
					vec = calc_subpixmax(corr, offset=np.r_[corr.shape]/2)
					plt.plot([shi[0]-shj[0]-vec[0]], [shi[1]-shj[1]-vec[1]], '*')
			raw_input()

if __name__ == "__main__":
	import sys
	from timeit import Timer
	import pylab as plt
	sys.exit(unittest.main())
