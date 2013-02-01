#!/usr/bin/env python
# encoding: utf-8
"""
@file test_fft.py
@brief Test libtim/fft.py library

@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120806

Testcases for fft.py library.
"""

from fft import *
import unittest
import pylab as plt

SHOWPLOTS=False

class TestEmbed(unittest.TestCase):
	def setUp(self):
		self.szlist = [(32, 32), (33, 33), (64, 65), (100, 201), (512, 512)]
		self.imlist = [np.random.random((sz)) for sz in self.szlist]

	def test0_try_scale2(self):
		"""Try to see if embed works for scale=2"""
		for im in self.imlist:
			emim = embed_data(im, direction=1, scale=2)
			ememim = embed_data(emim, direction=-1, scale=2)
			#print "test0_try_scale2(): im = %s, em = %s" % (str(im.shape), str(emim.shape))

	def test0_try_scale3(self):
		"""Try to see if embed works for scale=3"""
		for im in self.imlist:
			emim = embed_data(im, direction=1, scale=3)
			ememim = embed_data(emim, direction=-1, scale=3)
			#print "test0_try_scale3(): im = %s, em = %s" % (str(im.shape), str(emim.shape))

	def test0_try_scale3_3(self):
		"""Try to see if embed works for scale=3.3"""
		for im in self.imlist:
			emim = embed_data(im, direction=1, scale=3.3)
			ememim = embed_data(emim, direction=-1, scale=3.3)
			#print "test0_try_scale3_3(): im = %s, em = %s" % (str(im.shape), str(emim.shape))

	def test1_unity_scale2(self):
		"""Try to see if embed and disembedding is unity"""
		for im in self.imlist:
			emim = embed_data(im, direction=1, scale=2)
			ememim = embed_data(emim, direction=-1, scale=2)
			#print "test1_unity_scale2(): im = %s, em = %s" % (str(im.shape), str(emim.shape))
			self.assertTrue(np.allclose(ememim, im))

	def test1_unity_scale3(self):
		"""Try to see if embed and disembedding is unity"""
		for im in self.imlist:
			emim = embed_data(im, direction=1, scale=3)
			ememim = embed_data(emim, direction=-1, scale=3)
			#print "test1_unity_scale3(): im = %s, em = %s" % (str(im.shape), str(emim.shape))
			self.assertTrue(np.allclose(ememim, im))


class TestApodMask(unittest.TestCase):
	def setUp(self):
		"""Define some constants for apodisation mask test case"""
		self.sz = (257, 509, 7)
		self.szlist = [self.sz[:n+1] for n in xrange(len(self.sz))]
		self.wsz_l = [-0.0, -0.1, -0.3, -0.7, -1.0]
		self.wshp_l = ['hann', 'hamming', 'cosine', 'lanczos']

	# Prototype: mk_apod_mask(masksz, apodpos=None, apodsz=None, shape='rect', wsize=0.1, apod_f=lambda x: x**2.0):

	# Shallow data tests
	# Shallow function test
	def test1a_return_shape(self):
		"""Returned shape should be sane"""
		for sz in self.szlist:
			self.assertEqual(mk_apod_mask(sz).shape, sz, \
				"Returned mask shape unexpected")

	# Deep function tests
	def test3a_hamm_nonzero(self):
		"""Hamming windows should not have zeros anywhere if they span the whole mask."""
		sz = self.szlist[1]
		thismask = mk_apod_mask(sz, apod_f='hamming')
		self.assertTrue((thismask > 0).all(), \
			"Fullsize Hamming window should never reach zero.")

		thismask = mk_apod_mask(sz, apod_f='hamming', shape='circ')
		self.assertTrue((thismask == 0).any(), \
			"Hamming should reach zero for circular shapes.")

	def test3b_maxval(self):
		"""For a mask with a full windowing range, there should be one pixels equal to 1."""
		sz = self.szlist[1]
		for func in self.wshp_l:
			thismask = mk_apod_mask(sz, wsize=-1, apod_f=func)
			neq1 = (thismask == 1).sum()
			self.assertTrue(neq1 == 1, \
				"Full window size mask should give only 1 pixel eq to 1 (got %d pixels)" % neq1)

	def test3c_allzero(self):
		"""If apodsz==0, the mask should be all zeros"""
		sz = self.szlist[1]
		for func in self.wshp_l:
			thismask = mk_apod_mask(sz, apodsz=0, apod_f=func)
			nnonzero = (thismask != 0).sum()
			self.assertEqual(nnonzero, 0, \
				"Apod size 0 gives values != 0 (got %d nonzeros)" % nnonzero)

	def test3d_wsize(self):
		"""For wsize=a, the number of elements == 1 should be N.round(((N.r_[apodsz]) * (1+a) - 1)/2, 0)*2.+1 """
		sz = self.szlist[1]
		for wsz in self.wsz_l:
			for func in ['hann', 'hamming', 'cosine', 'lanczos']:
				thismask = mk_apod_mask(sz, apod_f=func, wsize=wsz)
				expvec = N.round(((N.r_[sz]) * (1+wsz) - 1)/2., 0)*2.+1
				expected = N.product(expvec)
				measured = (thismask >= 1.0).sum()
				self.assertAlmostEqual(expected, measured, 0,\
					msg="Unexpected number of number of elements equal to one. Expected %g, got %g for wsz=%g, wf=%s" % (expected, measured, wsz, func))

	def test3d_wsize0(self):
		"""For wsize=0, all windows shapes should be identical (except hamming which is nonzero at the edge)"""
		wsz = 0
		ignfunc = ["hamming"]
		for sz in self.szlist[:2]:
			refmask = mk_apod_mask(sz, apod_f=self.wshp_l[0], wsize=wsz)
			for func in self.wshp_l[1:]:
				if (func in ignfunc):
					continue
				thismask = mk_apod_mask(sz, apod_f=func, wsize=wsz)
				self.assertTrue(N.allclose(thismask, refmask), \
					"Windows size 0 mask for %s != %s" % (self.wshp_l[0], func))

	# Test illegal function calls
	def test4a_apodf_err(self):
		"""Test if illegal apod_f raises error"""

		for sz in self.szlist:
			with self.assertRaisesRegexp(ValueError, ".*apod_f.*not supported!"):
				mk_apod_mask(sz, apod_f="not a function")
		for sz in self.szlist:
			with self.assertRaisesRegexp(ValueError, ".*apod_f.*should be.*"):
				mk_apod_mask(sz, apod_f=[1])

	def test4b_shape_err(self):
		"""Test if illegal shape raises error"""

		for sz in self.szlist:
			with self.assertRaisesRegexp(ValueError, "<shape> should be.*"):
				mk_apod_mask(sz, shape="not a shape")
		for sz in self.szlist:
			with self.assertRaisesRegexp(ValueError, ".*should be a string!"):
				mk_apod_mask(sz, shape=1)

class PlotApodMask(unittest.TestCase):
	def setUp(self):
		"""Define some constants for apodisation mask test case"""
		self.sz = (257, 509, 7)
		self.szlist = [self.sz[:n+1] for n in xrange(len(self.sz))]
		self.wsz_l = [-0.0, -0.3, -1.0]
		self.wshp_l = ['hann', 'hamming', 'cosine', 'lanczos']

	# Display functions (if all else succeeded)
	def test0a_dummy(self):
		"""Dummy test"""
		print "This is PlotApodMask()"

	def test4a_plotmasks(self):
		"""Plot some default masks"""
		if (not SHOWPLOTS):
			continue
		print "Plotting default masks"
		for sz in self.szlist:
			thismask = mk_apod_mask(sz)
			if len(sz) == 1:
				plt.clf()
				plt.title("test4a_plotmasks 1d default")
				plt.plot(thismask)
				raw_input()
			elif len(sz) == 2:
				plt.clf()
				plt.title("test4a_plotmasks 2d default")
				plt.imshow(thismask, interpolation='nearest')
				plt.colorbar()
				raw_input()

	def test4b_plot_wsizes(self):
		"""Plot different 1-d window sizes and shapes"""
		print "Plot different 1-d window functions sizes and shapes"
		sz = self.szlist[0]
		for wsz in self.wsz_l:
			plt.clf()
			plt.title('Window size=%g, shapes=%s' % (wsz, str(self.wshp_l)))
			for func in self.wshp_l:
				thismask = mk_apod_mask(sz, apod_f=func, wsize=wsz)
				plt.plot(thismask, label=func)
			plt.legend()
			raw_input()

	def test4c_plot_apodsizes(self):
		"""Plot different 1-d window functions"""
		sz = self.szlist[0]
		for apodsz in [0, -1]:
			plt.clf()
			plt.title('Apod size=%g, shapes=%s' % (apodsz, str(self.wshp_l)))
			for func in self.wshp_l:
				thismask = mk_apod_mask(sz, apodsz=apodsz, apod_f=func)
				plt.plot(thismask, label=func)
			plt.legend()
			raw_input()

	def test4d_plot_2dcirc(self):
		"""Plot different 2-d circular windows with varying pos and size"""
		print "Plot different 2-d circular windows with varying pos and size"
		sz = self.szlist[1]
		plt.figure(1)
		plt.clf()
		plt.suptitle('2D circular masks')
		for i, apodpos in enumerate((-0.5, 100)):
			for j,apodsz in enumerate((-0.5, 20)):
				plt.subplot(2,2, i*2+j)
				thismask = mk_apod_mask(sz, apodpos, apodsz, wsize=-1, shape='circ')
				plt.imshow(thismask, interpolation='nearest')
				plt.title('pos=%g, size=%g' % (apodpos, apodsz))
		raw_input()

if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())
