#!/usr/bin/env python
# encoding: utf-8
"""
@file fft.py
@brief Utilities for Fourier transforms

@package libtim.fft
@brief Utilities for Fourier transforms
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120403

Package for some utilities for Fourier transforms
"""

#============================================================================
# Import libraries here
#============================================================================

import numpy as np
from collections import Iterable
import libtim as tim
from math import ceil, floor

#============================================================================
# Defines
#============================================================================

#============================================================================
# Routines
#============================================================================

def mk_apod_mask(masksz, apodpos=None, apodsz=None, shape='rect', wsize=-0.3, apod_f=lambda x: 0.5 * (1.0 - np.cos(np.pi*x))):
	"""
	Generate apodisation mask with custom size, shape, edge.

	The output array mask will be **masksz**, while the actual apodisation masked will **apodsz** big. The position of the mask is given with **apodpos**.

	**apodpos** defaults to the center, **apodsz** defaults to **masksz**

	**apodpos**, **apodsz** and **wsize** can either be given as fraction (if < 0) or as absolute number of pixels (if > 0). If these are given in int or float, the result will be square, if these are tuples, the size can be different in both dimensions.

	If **apodpos** or **apodsz** are fractional, they are relative to **masksz**. Fractional **wsize** is relative to **apodsz**.

	**apod_f** is the windowing function used. It can be a string (see list below), or a lambda function. In the latter case it should take one float coordinate between 1 and 0 as input and return the value of the window at that position.

	Some apodisation functions (for **apod_f**):
	- 'Hann': lambda x: 0.5 * (1.0 - np.cos(np.pi*x))
	- 'Hamming': lambda x: 0.54 - 0.46 *np.cos(np.pi*x)
	- '(Co)sine' window: lambda x: np.sin(np.pi*x*0.5)
	- 'Lanczos': lambda x: np.sinc(x-1.0)

	@param [in] masksz Size of the output array containing the apodisation mask
	@param [in] apodpos Position of the apodisation mask
	@param [in] apodsz Size of the apodisation mask
	@param [in] shape Apodisation mask shape, 'rect' or 'circular'
	@param [in] wsize Size of the apodisation window, i.e. the transition region to go from 0 to 1.
	@param [in] apod_f Apodisation function to use. Can be lambda function
	"""

	# Check apodpos and apodsz, if not set, use defaults
	if (apodpos == None):
		apodpos = tuple((np.r_[masksz]-1.)/2.)
	if (apodsz == None):
		apodsz = masksz

	apod_func = lambda x: x
	if (isinstance(apod_f, str)):
		apod_f = apod_f.lower()
		if (apod_f[:4] == 'hann'):
			apod_func = lambda x: 0.5 * (1.0 - np.cos(np.pi*x))
		elif (apod_f[:4] == 'hamm'):
			apod_func = lambda x: 0.54 - 0.46 *np.cos(np.pi*x)
		elif (apod_f[:3] == 'cos' or apod_f[:3] == 'sin'):
			apod_func = lambda x: np.sin(np.pi*x*0.5)
		elif (apod_f[:4] == 'lanc'):
			apod_func = lambda x: np.sinc(x-1.0)
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
		apodpos *= -np.r_[masksz]
	if (min(apodsz) < 0):
		apodsz *= -np.r_[masksz]
	if (min(wsize) < 0):
		wsize *= -np.r_[apodsz]

	# Generate base mask, which are (x,y) coordinates around the center
	mask = np.indices(masksz, dtype=np.float)

	# Center the mask around <apodpos> for any number of dimensions
	for (maski, posi) in zip(mask, apodpos):
		maski -= posi

	# If the mask shape is circular, calculate the radial distance from
	# <apodpos>
	if (shape == 'circ'):
		mask = np.array([np.sum(mask**2.0, 0)**0.5])

	# Scale the pixels such that there is only a band going from 1 to 0 between <masksz>-<wsize> and <masksz>
	for (maski, szi, wszi) in zip(mask, apodsz, wsize):
		# First take the negative absolute value of the mask, such that 0 is at the origin and the value goes down outward from where the mask should be.
		maski[:] = -np.abs(maski)
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
		return np.product(mask, 0)
	elif (shape == 'circ'):
		return (mask[0])

def descramble(data, direction=1):
	"""
	(de)scramble **data**, usually used for Fourier transform.

	'Scrambling' data means to swap around quadrant 1 with 3 and 2 with 4 in a data matrix. The effect is that the zero frequency is no longer at **data[0,0]** but in the middle of the matrix

	@param [in] data Data to (de)scramble
	@param [in] direction 1: scramble, -1: descramble
	@return (de)scrambled data
	"""

	for ax,rollvec in enumerate(np.r_[data.shape]/2):
		data = np.roll(data, direction*rollvec, ax)

	return data

def embed_data(indata, direction=1, scale=2):
	"""
	Embed **indata** in a zero-filled rectangular array **scale** times as big as **indata**.

	To prevent wrapping artifacts in Fourier analysis, this function can  embed data in a zero-filled rectangular array of twice the size.

	If **direction** = 1, **indata** will be embedded, if **direction** = -1, it will be dis-embedded.

	@param [in] indata Data to embed
	@param [in] direction 1: embed, -1: dis-embed
	@param [in] scale Size of embedding matrix wrt **indata**
	@return (dis)-embedded data, either scale*indata.shape or 1/scale*indata.shape
	"""
	
	s = np.r_[indata.shape]

	if (direction == 1):
		# Generate empty array
		retdat = np.zeros(np.r_[s]*scale, dtype=indata.dtype)
		# These slices denote the central region where <indata> will go
		slice0 = slice(retdat.shape[0]/2 - floor(s[0]/2.0), 
						retdat.shape[0]/2 + ceil(s[0]/2.0))
		slice1 = slice(retdat.shape[1]/2 - floor(s[1]/2.0), 
						retdat.shape[1]/2 + ceil(s[1]/2.0))

		#
		# Insert the data and return it
		retdat[slice0, slice1] = indata
		return retdat
	else:
		# These slices give the central area of the data
		slice0 = slice(s[0]/2 - floor(s[0]/(2.*scale)), 
						s[0]/2 + ceil(s[0]/(2.*scale)))
		slice1 = slice(s[1]/2 - floor(s[1]/(2.*scale)), 
						s[1]/2 + ceil(s[1]/(2.*scale)))

		#tim.shell()
		# Slice out the center and return it
		return indata[slice0, slice1]
