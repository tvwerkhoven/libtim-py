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
import libtim as tim
import libtim.zern

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

def calc_slope(im, slopes=None):
	"""
	Calculate 2D slope of **im**, to be used to calculate unit Zernike 
	influence on SHWFS. If **slopes** is given, use that (2, N) matrix for 
	fitting, otherwise generate and pseudo-invert slopes ourselves.

	@param [in] im Image to fit slopes to
	@param [in] slopes Pre-computed inverted slope matrix to fit with

	@return Tuple of (influence matrix, slope matrix, Zernike basis used)
	"""

	if (slopes == None):
		slopes = (np.indices(im.shape, dtype=float)/(np.r_[im.shape].reshape(-1,1,1))).reshape(2,-1)
		slopes2 = np.vstack([slopes, slopes[0]*0+1])
		slopes = np.linalg.pinv(slopes)

	return np.dot(im.reshape(1,-1)-np.mean(im), slopes).ravel()[:2]

	# This doens't work, why? Some normalisation error?
	slope0, slope1 = (np.indices(im.shape, dtype=float)/(np.r_[im.shape].reshape(-1,1,1)))
	coeff0 = np.sum(im.ravel() * slope0.ravel())/np.sum(slope0)
	coeff1 = np.sum(im.ravel() * slope1.ravel())/np.sum(slope1)
	print coeff0, coeff1
	return coeff0, coeff1

def calc_zern_infmat(subaps, nzern=10, zernrad=-1.0, check=True, focus=1.0, wavelen=1.0, subapsize=1.0, pixsize=1.0):
	"""
	Given a subaperture array pattern, calculate a matrix that converts 
	image shift vectors in pixel to Zernike amplitudes.

	@param [in] subaps List of subapertures formatted as (low0, high0, low1, high1)
	@param [in] nzern Number of Zernike modes to model
	@param [in] zernrad Radius of the aperture to use. If less negative, used as fraction **-zernrad**, otherwise used as radius in pixels.
	@param [in] check Perform basic sanity checks
	@param [in] focus Focal length of MLA (in meter)
	@param [in] wavelen Wavelength used for SHWFS (in meter)
	@param [in] subapsize Size of single microlens (in meter)
	@param [in] pixsize Pixel size (in meter)
	"""

	# Conversion factor from Zernike radians to pixels: F*λ/2π/d/pix_pitch
	sfac = focus * wavelen / (2*np.pi * subapsize * pixsize)

	# Geometry: offset between subap pattern and Zernike modes
	sasize = np.median(subaps[:,1::2] - subaps[:,::2], axis=0)
	
	pattcent = np.mean(subaps[:,::2], axis=0).astype(int)
	pattrad = np.max(np.max(subaps[:, 1::2], 0) - np.min(subaps[:, ::2], 0))

	if (zernrad < 0):
		rad = int(pattrad*-zernrad+0.5)
	else:
		rad = int(zernrad+0.5)

	saoffs = -pattcent + np.r_[ [rad, rad] ]

	zbasis = tim.zern.calc_zern_basis(nzern, rad)

	# Check coordinates are sane
	if (check):
		crop_coords = np.r_[ [[(subap[0]+saoffs[0], subap[2]+saoffs[1]) for subap in subaps] for zbase in zbasis['modes']] ]
		assert np.max(crop_coords) < 2*rad
		assert np.min(crop_coords) > 0

	# Initialize fit matrix
	slopes = (np.indices(sasize, dtype=float)/(np.r_[sasize].reshape(-1,1,1))).reshape(2,-1)
	slopesi = np.linalg.pinv(slopes)

	zernslopes = np.r_[ [[calc_slope(zbase[subap[0]+saoffs[0]:subap[1]+saoffs[0], subap[2]+saoffs[1]:subap[3]+saoffs[1]], slopes=slopesi) for subap in subaps] for zbase in zbasis['modes']] ].reshape(nzern, -1)

	# Construct inverted matrix using 95% singular value
	U, s, Vh = np.linalg.svd(zernslopes*sfac, full_matrices=False)

	nvec = np.argwhere(s.cumsum()/s.sum() > 0.95)[0][0]
	s[nvec:] = np.inf
	return np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T)), zernslopes*sfac, zbasis


def find_mla_grid(wfsimg, size, clipsize=None, minif=0.6, nmax=-1, copy=True, method='bounds', sort=False):
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
	@param [in] sort Sort subaperture coordinates
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

	subap_grid = np.r_[subap_grid]

	if (sort):
		# Sort by increasing pixel
		sortidx = np.argsort( (subap_grid[:,0]/size[0]).astype(int)*wfsimg.shape[0] + subap_grid[:,2])
	else:
		sortidx = slice(None)
	
	return subap_grid[sortidx]

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

def locate_acts(infmat, subappos, nsubap=20, weigh=True, verb=0):
	"""
	Given the influence matrix and the sub aperture positions, find the 
	actuator positions using the intersection of the influence direction of 
	an actuator for each sub aperture.

	**infmat** should be shaped (nact, nsubap, 2)
	**subappos** should be shaped (nsubap, 2)
	**nsubap** determines how many subaps will be used per actuator

	@param [in] infmat Influence matrix shaped (nact, nsubap, 2)
	@param [in] subappos Subap position vector shaped (nsubap, 2)
	@param [in] nsubap Number of sub apertures to use for fitting the actuator positions
	@param [in] weigh Use shift vector norm as weight when fitting or not
	@return (nact, 2) vector with the actuator positions
	"""

	# For each actuator, find the **nsubap** most influential subapertures 
	# and the influence
	actposl = []
	for actid, actinf in enumerate(infmat):
		# For this actuator, get **nsubap** most influential subapertures
		actinf_v = (actinf**2.0).sum(1)**0.5
		subaps_idx = np.argsort(actinf_v)[-nsubap:]
	
		if (verb > 2):
			# Plot subaperture influences
			meaninf = actinf_v[subaps_idx].mean()
			import pylab as plt
			plt.figure(); plt.clf()
			plt.title("actid=%d, subaps=%s, meaninf=%g" % (actid, str(subaps_idx), meaninf))
			q = plt.quiver(subappos[subaps_idx, 1], subappos[subaps_idx, 0], actinf[subaps_idx, 1], actinf[subaps_idx, 0], angles='xy')

		actpos = calc_intersect(posvecs=subappos[subaps_idx], dvecs=actinf[subaps_idx], weigh=weigh)
		
		if (verb > 2):
			plt.plot([actpos[1]],
					[actpos[0]],  'x')
			raw_input("Press any key to continue...")
			plt.close()
		
		actposl.append(actpos)
	
	return np.r_[actposl]

def calc_intersect(posvecs, dvecs, weigh=True):
	"""
	Given a matrix of (N, 2) position vectors **posvecs** and a similarly 
	shaped matrix of distance vectors **dvecs**, calculate the intersection 
	of these. If **weigh** is True, use the length of **dvecs** as weight.

	Cost function for 1 spot:

		d(x, (p,n)) = (x-p)^T (nn)^T (x-p)
	
	Cost for n spots, weight w_i

		d_i(x, (p,n)) = sqrt(w_i) (x-p_i)^T (n_i n_i)^T (x-p_i)

	Minimize cost function with x
	
	References
	- https://en.wikipedia.org/wiki/Line-line_intersection
	- https://en.wikipedia.org/wiki/Least_squares
	- https://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29

	@param [in] posvecs List of x,y position vectors, (N, 2)
	@param [in] dvecs List of x,y displacement vectors, (N, 2)
	@param [in] weigh Use norm of **dvecs** as weight during fitting
	@return Tuple of (best fit intersection coordinate, standard deviation)
	"""

	if len(posvecs) != len(dvecs): 
		raise ValueError("Unequal length **posvecs** and **dvecs** encountered")
	if (len(posvecs) < 2):
		raise ValueError("Need at least two data points to fit")

	normals = np.dot(dvecs, np.r_[ [[0,-1],[1,0]] ])
	normals /= ((normals**2.0).sum())**0.5

	if (weigh):
		weights = (dvecs**2.0).sum(1)**0.5 / (dvecs**2.0).sum()**0.5
	else:
		weights = np.ones(dvecs.shape[0])

	# Calculate minimum of cost function (see Wikipedia)
	s1 = s2 = 0
	for norm, loc, w in zip(normals, posvecs, weights):
		norm.shape = (1,-1)
		s1 += w * np.dot(loc, np.dot(norm.T, norm))
		s2 += w * np.dot(norm.T, norm)

	return np.dot(s1, np.linalg.pinv(s2))


if __name__ == "__main__":
	sys.exit(unittest.main())
