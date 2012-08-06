#!/usr/bin/env python
# encoding: utf-8
"""
@file test_im.py
@brief Test libtim.im

@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120806

Test libtim.im
"""

from im import *
import unittest
import pylab as plt


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
        
		print "removing ", fpaths
        
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
	sys.exit(unittest.main())
