#!/usr/bin/env python
# encoding: utf-8
"""
@file shwfs.py
@brief Shack-Hartmann wavefront sensor analysis tools

@package libtim.shwfs
@brief Shack-Hartmann wavefront sensor analysis tools
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120620

This module provides functions to analyze Shack-Hartmann wavefront sensor data
"""

#=============================================================================
# Import libraries here
#=============================================================================

import sys
import os
import numpy as N
import unittest

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def calc_cog(img):
	"""
	Calculate center of gravity
	
	@param [in] img
	@return Sub-pixel coordinate of center of gravity (c0, c1)
	"""
	
	c0 = (img.sum(0) * N.arange(img.shape[0])).sum()/img.sum()
	c1 = (img.sum(1) * N.arange(img.shape[1])).sum()/img.sum()
	return (c0, c1)

def find_mla_grid(wfsimg, size, minif=0.6, nmax=-1):
	"""
	Given a wavefront sensor image, find a grid of subapertures (sa) that 
	match the image.
	
	@param [in] wfsimg Image to analyze
	@param [in] size Subaperture size [pixels]
	@param [in] minif Minimimum intensity a pixel has to have to consider it as a new subaperture, as fraction of the maximum intensity.
	@param [in] nmax Maximum number of subapertures to search for
	"""
	
	if (size[0] > wfsimg.shape[0] or size[1] > wfsimg.shape[1]):
		raise ValueError("Subapertures sizes larger than the soure image resolution don't make sense.")
	
	# Copy image because we destroy it
	img = wfsimg.copy()
	
	# Minimum intensity to consider
	min_int = wfsimg.max() * minif
	
	subap_grid = []
	
	while (True):
		# The current maximum intesnity is at:
		currmax = img.max()
		maxidx = N.argwhere(img == currmax)[0]
		
		if (currmax < min_int):
			break
		
		# Add this subaperture
		newsa = (maxidx[0] - size[0]/2, 
				maxidx[0] + size[0]/2,
				maxidx[1] - size[1]/2,
				maxidx[1] + size[1]/2)
		subap_grid.append(newsa)
		
		# Clear out subaperture so we don't add it again
		img[newsa[0]:newsa[1], newsa[2]:newsa[3]] = img.min()
		
		if (nmax > 0 and len(subap_grid) >= nmax):
			break
	
	return subap_grid

def calc_subap_grid(rad, size, pitch, shape='circular', xoff=[0,0.5], disp=(0,0), overlap=1.0, scl=1.0):
	"""
	Generate subaperture (sa) positions for a given configuration.

	@param rad radius of the sa pattern (before scaling) [pixels]
	@param size size of the sa's [pixels]
	@param pitch pitch of the sa's [pixels]
	@param shape shape of the sa pattern ['circular' or 'square']
	@param xoff the horizontal position offset of odd rows (in units of 'size')
	@param disp global displacement of the sa positions [pixels]
	@param overlap how much overlap should the subap have to be included (0=complete, 1=any overlap)
	@param scl global scaling of the sa positions [pixels]

	@return (<# of subaps>, <lower positions>, <center positions>, <size>)

	Raises ValueError if shape is unknown and RuntimeError if no subapertures
	we found using the specified configuration.
	"""

	rad = N.array(rad)
	size = N.array(size)
	disp = N.array(disp)
	pitch = N.array(pitch)
	xoff = N.array(xoff)

	# (half) width and height of the subaperture array
	sa_arr = (N.ceil(rad/pitch)+1).astype(int)
	# Init empty list to store positions
	pos = []
	# Loop over all possible subapertures and see if they fit inside the
	# aperture shape. We loop y from positive to negative (top to bottom
	# in image coordinates) and x from negative to positive (left to
	# right)
	for say in range(sa_arr[1], -sa_arr[1]-1, -1):
		for sax in range(-sa_arr[0], sa_arr[0]+1, 1):
			# Centroid coordinate for this possible subaperture is:
			sac = [sax, say] * pitch

			# 'say % 2' gives 0 for even rows and 1 for odd rows. Use this
			# to apply a row-offset to even and odd rows
			# If we're in an odd row, check saccdoddoff
			sac[0] -= xoff[say % 2] * pitch[0]

			# Check if we're in the apterture bounds, and store the subapt
			# position in that case
			if (shape == 'circular'):
				if (sum((abs(sac)+(overlap*size/2.0)/2.0)**2) < rad**2).all():
					pos.append(sac)
			elif shape == 'square':
				if (abs(sac)+(overlap*size/2.0) < rad).all():
					pos.append(sac)
			else:
				raise ValueError("Unknown aperture shape '%s'" % (apts))

	if (len(pos) <= 0):
		raise RuntimeError("Didn't find any subapertures for this configuration.")

	# Apply scaling and displacement to the pattern before returning
	# NB: pos gives the *centroid* position of the subapertures here
	cpos = (N.array(pos) * scl) + disp

	# Convert symmetric centroid positions to origin positions:
	llpos = cpos - size/2.0

	nsa = len(llpos)

	return (nsa, llpos, cpos, size)

### TEST SUITES

# arr = N.zeros((16,16))
# arr[10,10] = 1000
# arr[11,11] = 1000
# print tim.shwfs.calc_cog(arr)
	
if __name__ == "__main__":
	sys.exit(unittest.main())
