#!/usr/bin/env python
# encoding: utf-8
"""
Some utilities for Fourier transforming data
"""

##  @file fft.py
# @author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
# @date 20120403
#
# Created by Tim van Werkhoven on 2012-04-03.
# Copyright (c) 2012 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
#
# This file is licensed under the Creative Commons Attribution-Share Alike
# license versions 3.0 or higher, see
# http://creativecommons.org/licenses/by-sa/3.0/

## @package fft
# @brief Utilities for Fourier transforms
# @author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
# @date 20120403
#
# Some utilities for Fourier transforms

#=============================================================================
# Import libraries here
#=============================================================================

import numpy as N
from collections import Iterable

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def mk_apod_mask(masksz, apodpos=None, apodsz=None, shape='rect', wsize=-0.3, apod_f=lambda x: 0.5 * (1.0 - N.cos(N.pi*x))):
	"""
	Generate apodisation mask. The returned array will have size <masksz>, with the apodisation mask centered at <apodpos> (element coordinates) with size <apodsz>. <apodpos> defaults to the center, <apodsz> defaults to <masksz>

	<shape> should be either 'rectangle' or 'circular'.
	<wsize> is the size of the window used.
	<apod_f> is the windowing function used. It should take one float coordinate between 1 and 0 as input and return the value of the window at that position.

	<apodpos>, <wsize> and <apodsz> can either as fraction (if < 0) or as absolute number of pixels (if > 0). Both can either be one int or float such that the same (fractional) size will be used in all dimensions, or it can be a tuple with a size for each dimension of <masksz>. In the latter case, all elements should either be absolute or fractional.

	When <apodsz> is fractional, it is relative to <masksz>. Fractional <wsize>'s are relative to <masksz>.

	Some apodisation functions:
	* Hann: lambda x: 0.5 * (1.0 - N.cos(N.pi*x))
	* Hamming: lambda x: 0.54 - 0.46 *N.cos(N.pi*x)
	* (Co)sine window: lambda x: N.sin(N.pi*x*0.5)
	* Lanczos: lambda x: N.sinc(x-1.0)
	"""

	# Check apodpos and apodsz, if not set, use defaults
	if (apodpos == None):
		apodpos = tuple((N.r_[masksz]-1.)/2.)
	if (apodsz == None):
		apodsz = masksz

	apod_func = lambda x: x
	if (isinstance(apod_f, str)):
		apod_f = apod_f.lower()
		if (apod_f[:4] == 'hann'):
			apod_func = lambda x: 0.5 * (1.0 - N.cos(N.pi*x))
		elif (apod_f[:4] == 'hamm'):
			apod_func = lambda x: 0.54 - 0.46 *N.cos(N.pi*x)
		elif (apod_f == 'cosine' or apod_f == 'sine'):
			apod_func = lambda x: N.sin(N.pi*x*0.5)
		elif (apod_f == 'lanczos'):
			apod_func = lambda x: N.sinc(x-1.0)
		else:
			raise ValueError("<apod_f> not supported!")
	elif (hasattr(apod_f, "__call__")):
		apod_func = apod_f
	else:
		raise ValueError("<apod_f> should be a string or callable!")

	# Mask size <masksz> should be iterable (like a list or tuple)
	if (not isinstance(masksz, Iterable)):
		raise TypeError("<masksz> should be iterable")
	if (min(masksz) < 1):
		raise ValueError("All mask size <masksz> dimensions should be >= 1")

	# Only the first 4 letters are significant.
	try:
		shape = shape[:4]
	except:
		raise ValueError("<shape> should be a string!")

	# Check if shape is legal
	if (shape not in ('rect', 'circ')):
		raise ValueError("<shape> should be 'rectangle' or 'circle'")

	# Check if apodpos, apodsz and wsize are legal. They should either be a
	# scalar (i.e. non-iterable) or the same length as <masksz> (which is iterable). Also, if apodpos, apodsz or wsize are just one int or float, repeat them for each dimension.
	if (isinstance(apodpos, Iterable) and len(apodpos) != len(masksz)):
		raise TypeError("<apodpos> should be either 1 element per dimension or 1 in total.")
	elif (not isinstance(apodpos, Iterable)):
		apodpos = (apodpos,) * len(masksz)

	if (isinstance(apodsz, Iterable) and len(apodsz) != len(masksz)):
		raise TypeError("<apodsz> should be either 1 element per dimension or 1 in total.")
	elif (not isinstance(apodsz, Iterable)):
		apodsz = (apodsz,) * len(masksz)

	if (isinstance(wsize, Iterable) and len(wsize) != len(masksz)):
		raise TypeError("<wsize> should be either 1 element per dimension or 1 in total.")
	elif (not isinstance(wsize, Iterable)):
		wsize = (wsize,) * len(masksz)

	# If apodsz or wsize are fractional, calculate the absolute size.
	if (min(apodpos) < 0):
		apodpos *= -N.r_[masksz]
	if (min(apodsz) < 0):
		apodsz *= -N.r_[masksz]
	if (min(wsize) < 0):
		wsize *= -N.r_[apodsz]

	# Generate base mask, which are (x,y) coordinates around the center
	mask = N.indices(masksz, dtype=N.float)

	# Center the mask around <apodpos> for any number of dimensions
	for (maski, posi) in zip(mask, apodpos):
		maski -= posi

	# If the mask shape is circular, calculate the radial distance from
	# <apodpos>
	if (shape == 'circ'):
		mask = N.array([N.sum(mask**2.0, 0)**0.5])

	# Scale the pixels such that there is only a band going from 1 to 0 between <masksz>-<wsize> and <masksz>
	for (maski, szi, wszi) in zip(mask, apodsz, wsize):
		# First take the negative absolute value of the mask, such that 0 is at the origin and the value goes down outward from where the mask should be.
		maski[:] = -N.abs(maski)
		# Next, add the radius of the apodisation mask size to the values, such that the outside edge of the requested mask is exactly zero.
		# TODO Should this be (szi-1)/2 or (szi)/2?
		maski += (szi)/2.
		# Now divide the mask by the windowing area inside the apod. mask, such that the inner edge of the mask is 1.
		if (wszi != 0):
			maski /= wszi/2.
		else:
			maski /= 0.001
		# Store masks for inside and outside the mask area
		inmask = maski > 1
		outmask = maski <= 0
		# Apply function to all data
		maski[:] = apod_func(maski[:])
		# Now everything higher than 1 is inside the mask, and smaller than 0 is outside the mask. Clip these values to (0,1)
		maski[inmask] = 1
		maski[outmask] = 0

	# Apply apodisation function to all elements, and multiply
	if (shape == 'rect'):
		return N.product(mask, 0)
	elif (shape == 'circ'):
		return (mask[0])

def descramble(data, direction=1):
	"""(de)scramble data, usually used for Fourier transform data"""

	for ax,rollvec in enumerate(N.r_[data.shape]/2):
		data = N.roll(data, direction*rollvec, ax)

	return data

def embed_data(indata, direction=1):
	"""
	Embed <indata> in a zero-filled rectangular array of twice the size for
	Fourier analysis.

	If <direction> is 1, <indata> will be embedded, if it is -1, it will be
	dis-embedded.
	"""

	if (direction == 1):
		# Generate empty array
		retdat = N.zeros(N.r_[indata.shape]*2)
		# These slices denote the central region where <indata> will go
		slice0 = slice(indata.shape[0]/2, 3*indata.shape[0]/2)
		slice1 = slice(indata.shape[1]/2, 3*indata.shape[1]/2)

		# Insert the data and return it
		retdat[slice0, slice1] = indata
		return retdat
	else:
		# These slices give the central area of the data
		slice0 = slice(indata.shape[0]/4, 3*indata.shape[0]/4)
		slice1 = slice(indata.shape[1]/4, 3*indata.shape[1]/4)

		# Slice out the center and return it
		return indata[slice0, slice1]


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
	import pylab as plt
	import unittest
	sys.exit(unittest.main())
