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

#==========================================================================
# Import libraries here
#==========================================================================

import sys
import os
import numpy as np
import unittest

#==========================================================================
# Defines
#==========================================================================

#==========================================================================
# Routines
#==========================================================================

def calc_cog(img, clip=0, clipf=None, index=False):
	"""
	Calculate center of gravity for a given 1, 2 or 3-dimensional array 
	**img**, optionally thresholding the data at **clip** first.

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

    Data can be thresholded if either clip or clipf is given. If clip is 
    given, use img-clip for CoG calculations. Else if clipf is given, use 
    img-clipf(img) for CoG. After data subtraction, data is clipped between 
    the original 0 and img.max(). When using thresholding, **img** 
    will be copied.

	N.B. Because of data ordering, dimension 0 (1) is **not** the x-axis 
	(y-axis) when plotting!
	
	@param [in] img input data, as ndarray
	@param [in] index If True, return CoG in pixel **index** coordinate, otherwise return pixel **center** coordinate.
	@param [in] clip Subtract this value from the data 
	@param [in] clipf Subtract clipf(img) from the data (if no clip)
	@return Sub-pixel coordinate of center of gravity ordered by data dimension (c0, c1)
	"""

	if (clip > 0):
		img = np.clip(img.copy() - clip, 0, img.max())
	elif (clipf != None):
		img = np.clip(img.copy() - clipf(img), 0, img.max())

	ims = img.sum()

	off = 0.5
	if (index): off = 0

	if (img.ndim == 1):
		# For dimension 0, sum all but dimension 0 (which in 2D is only dim 1)
		return ((img * np.arange(img.shape[0])).sum()/ims) + off
	elif (img.ndim == 2):
		c0 = (img.sum(1) * np.arange(img.shape[0])).sum()/ims
		c1 = (img.sum(0) * np.arange(img.shape[1])).sum()/ims
		return np.r_[c0, c1] + off
	elif (img.ndim == 3):
		c0 = (img.sum(2).sum(1) * np.arange(img.shape[0])).sum()/ims
		c1 = (img.sum(2).sum(0) * np.arange(img.shape[1])).sum()/ims
		c2 = (img.sum(1).sum(0) * np.arange(img.shape[2])).sum()/ims
		return np.r_[c0, c1, c2] + off
	else:
		raise RuntimeError("More than 3 dimensional data not supported!")

def find_mla_grid(wfsimg, size, clipsize=None, minif=0.6, nmax=-1, copy=True, method='bounds'):
	"""
	Given a Shack-hartmann wave front sensor image, find a grid of 
	subapertures (sa) of approximately **size** big.

	The image will be destroyed during analysis, unless **copy** is True 
	(default)
	
	@param [in] wfsimg Image to analyze
	@param [in] size Subaperture size [pixels]
	@param [in] size Size to clip out when searching for spots [pixels]
	@param [in] minif Minimimum pixel value to have to consider it as a new subaperture, as fraction of the global maximum.
	@param [in] nmax Maximum number of subapertures to search for (-1 for no max)
	@param [in] copy Copy image before modifying
	@param [in] method Coordinate format to return, either 'bounds' or 'center'
	@return list of subaperture coordinates

	Raises ValueError if subapertures size is larger than source image.
	"""
	
	if (size[0] > wfsimg.shape[0] or size[1] > wfsimg.shape[1]):
		raise ValueError("Subapertures sizes larger than the source image resolution don't make sense.")
	
	cs = clipsize
	if (cs == None): cs = size

	# Copy image because we destroy it
	if (copy): wfsimg = wfsimg.copy()

	# Minimum intensity to consider
	min_int = wfsimg.max() * minif
	
	subap_grid = []
	
	while (True):
		# The current maximum intensity is at:
		currmax = wfsimg.max()
		p = np.argwhere(wfsimg == currmax)[0]
		
		if (currmax < min_int):
			break
		
		# Add this subaperture, either by bounds or central coordinate
		if (method == 'bounds'):
			newsa = (p[0] - size[0]/2, 
					p[0] + size[0]/2,
					p[1] - size[1]/2,
					p[1] + size[1]/2)
		else:
			newsa = tuple(p)
		subap_grid.append(newsa)
		
		# Clear out this subaperture so we don't add it again

		wfsimg[p[0]-cs[0]/2:p[0]+cs[0]/2, p[1]-cs[1]/2:p[1]+cs[1]/2] = wfsimg.min()
		
		if (nmax > 0 and len(subap_grid) >= nmax):
			break
	
	return subap_grid

def calc_subap_grid(rad, size, pitch, shape='circular', xoff=(0, 0.5), disp=(0,0), overlap=1.0, scl=1.0):
	"""
	Generate regular subaperture (sa) positions for a given configuration.

	@param rad radius of the sa pattern (before scaling) [pixels]
	@param size size of the sa's [pixels]
	@param pitch pitch of the sa's [pixels]
	@param shape shape of the sa pattern ['circular' or 'square']
	@param xoff the horizontal position offset of odd rows (in units of 'size'), set to 0.5 for hexagonal grids
	@param disp global displacement of the sa positions [pixels]
	@param overlap how much overlap should the subap have to be included (0=complete, 1=any overlap)
	@param scl global scaling of the sa positions [pixels]

	@return (<# of subaps>, <lower positions>, <center positions>, <size>)

	Raises ValueError if shape is unknown and RuntimeError if no subapertures
	we found using the specified configuration.
	"""

	# Convert to arrays just to be sure
	rad = np.array(rad)
	size = np.array(size)
	disp = np.array(disp)
	pitch = np.array(pitch)
	xoff = np.array(xoff)

	# (half) width and height of the subaperture array
	sa_arr = (np.ceil(rad/pitch)+1).astype(int)
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
	cpos = (np.array(pos) * scl) + disp

	# Convert symmetric centroid positions to origin positions:
	llpos = cpos - size/2.0

	nsa = len(llpos)

	return (nsa, llpos, cpos, size)

if __name__ == "__main__":
	sys.exit(unittest.main())
