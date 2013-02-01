#!/usr/bin/env python
# encoding: utf-8
"""
@file test_shwfs.py
@brief Test libtim/shwfs.py library

@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20130130

Testcases for shwfs.py library.
"""

from shwfs import *
import unittest
import pylab as plt

class TestCoG(unittest.TestCase):
	def setUp(self):
		self.szlist = [(32, 32), (33, 33), (64, 65), (100, 201), (512, 512)]
		self.rndimlist = [np.random.random((sz)) for sz in self.szlist]
		self.imlist = [np.zeros((sz)) for sz in self.szlist]
		self.maxpos = [(15, 5), (17, 17), (63, 0), (50, 50), (50, 50)]
		for im, rim, p in zip(self.imlist, self.rndimlist, self.maxpos):
			im[p] = 10
			# The peak should be ten times higher than the background noise to get <0.1 pixel CoG noise (on average this will give <0.05 pixel noise)
			rim[p] = np.product(np.r_[rim.shape])*10
	
	def test0_try_calc_cog(self):
		"""Try to see if calc_cog"""
		for im in self.imlist:
			cog = calc_cog(im)
	
	def test1a_check_calc_cog(self):
		"""Try to see if calc_cog output is  sane"""
		for im, p in zip(self.imlist, self.maxpos):
			cog = calc_cog(im)
			self.assertEqual(cog, p)

	def test1b_check_calc_cog(self):
		"""Try to see if calc_cog output is  sane"""
		for rim, p in zip(self.rndimlist, self.maxpos):
			cog = calc_cog(rim)
			self.assertLess((p[0]-cog[0]) + (p[1]-cog[1]), 0.2)

if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())
