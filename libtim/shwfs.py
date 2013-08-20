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
try:
	import pyfftw
except:
	pass

#==========================================================================
# Defines
#==========================================================================

#==========================================================================
# Routines
#==========================================================================

def calc_cog(img, clip=0, clipf=None, index=False, clipcheck=True):
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
	@param [in] clipcheck Check if clipping still gives some data, raise if not
	@return Sub-pixel coordinate of center of gravity ordered by data dimension (c0, c1)
	"""

	# When clipping, always (up)cast to float in case we have some unsigned 
	# data
	if (clip > 0):
		img = np.clip(img.copy().astype(float) - clip, 0, img.max())
		if (clipcheck):
			assert img.max() > 0, "clipped image has only zeros"
	elif (clipf != None):
		img = np.clip(img.copy().astype(float) - clipf(img), 0, img.max())
		if (clipcheck):
			assert img.max() > 0, "clipped image has only zeros"

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

def calc_slope(im, slopesi=None):
	"""
	Calculate 2D slope of **im**, to be used to calculate unit Zernike 
	influence on SHWFS. If **slopesi** is given, use that (2, N) matrix for 
	fitting, otherwise generate and pseudo-invert slopes ourselves.

	@param [in] im Image to fit slopes to
	@param [in] slopesi Pre-computed **inverted** slope matrix to fit with, leave empty to auto-calculate.
	@return Tuple of (slope axis 0, slope axis 1)
	"""

	if (slopesi == None):
		slopes = (np.indices(im.shape, dtype=float)/(np.r_[im.shape].reshape(-1,1,1))).reshape(2,-1)
		slopes2 = np.vstack([slopes, slopes[0]*0+1])
		slopesi = np.linalg.pinv(slopes2)

	return np.dot(im.reshape(1,-1), slopesi).ravel()[:2]

def calc_zern_infmat(subaps, nzern=10, zerncntr=None, zernrad=-1.0, singval=1.0, check=True, focus=1.0, wavelen=1.0, subapsize=1.0, pixsize=1.0, verb=0):
	"""
	Given a sub aperture array pattern, calculate a matrix that converts 
	image shift vectors in pixel to Zernike amplitudes, and its inverse.

	The parameters **focus**, **wavelen**, **subapsize** and **pixsize** are 
	used for absolute calibration. If these are provided, the shifts in 
	pixel are translated to Zernike amplitudes where amplitude has unit 
	variance, i.e. the normalisation used by Noll (1976). These parameters
	should all be in meters.

	The data returned is a tuple of the following:

	1. Matrix to compute Zernike modes from image shifts
	2. Matrix to image shifts from Zernike amplitudes
	3. The set of Zernike polynomials used, from tim.zern.calc_zern_basis()
	4. The extent of the Zernike basis in units of **subaps** which can be used as extent keyword to imshow() when plotting **subaps**.

	To calculate the above mentioned matrices, we measure the x,y-slope of 
	all Zernike modes over all sub apertures, giving a matrix 
	`zernslopes_mat` that converts slopes for each Zernike matrix:

		subap_slopes_vec = zernslopes_mat . zern_amp_vec

	We multiply these slopes in radians/subaperture by 

		sfac = F * λ/2π/d/pix_pitch

	to obtain pixel shifts inside the sub images. We then have

		subap_shift_vec = sfac * zernslopes_mat . zern_amp_vec

	to get the inverse relation, we invert `zernslopes_mat`, giving:

		zern_amp_vec = (sfac * zernslopes_mat)^-1 . subap_shift_vec
		zern_amp_vec = zern_inv_mat . subap_shift_vec

	@param [in] subaps List of subapertures formatted as (low0, high0, low1, high1)
	@param [in] nzern Number of Zernike modes to model
	@param [in] zerncntr Coordinate to center Zernike around. If None, use center of **subaps**
	@param [in] zernrad Radius of the aperture to use. If negative, used as fraction **-zernrad**, otherwise used as radius in pixels.
	@param [in] singval Percentage of singular values to take into account when inverting the matrix
	@param [in] check Perform basic sanity checks
	@param [in] focus Focal length of MLA (in meter)
	@param [in] wavelen Wavelength used for SHWFS (in meter)
	@param [in] subapsize Size of single microlens (in meter)
	@param [in] pixsize Pixel size (in meter)
	@param [in] verb Show plots indicating fit geometry
	@return Tuple of (shift to Zernike matrix, Zernike to shift matrix, Zernike polynomials used, Zernike base shape in units of **subaps**)
	"""

	# Conversion factor from Zernike radians to pixels: F*λ/2π/d/pix_pitch
	sfac = focus * wavelen / (2*np.pi * subapsize * pixsize)

	# Geometry: offset between subap pattern and Zernike modes
	sasize = np.median(subaps[:,1::2] - subaps[:,::2], axis=0)
	if (zerncntr == None):
		zerncntr = np.mean(subaps[:,::2], axis=0).astype(int)

	if (zernrad < 0):
		pattrad = np.max(np.max(subaps[:, 1::2], 0) - np.min(subaps[:, ::2], 0))/2.0
		rad = int((pattrad*-zernrad)+0.5)
	else:
		rad = int(zernrad+0.5)
	saoffs = -zerncntr + np.r_[ [rad, rad] ]

	extent = zerncntr[1]-rad, zerncntr[1]+rad, zerncntr[0]-rad, zerncntr[0]+rad
	zbasis = tim.zern.calc_zern_basis(nzern, rad, modestart=2)

	# Check coordinates are sane
	if (check):
		crop_coords = np.r_[ [[(subap[0]+saoffs[0], subap[2]+saoffs[1]) for subap in subaps] for zbase in zbasis['modes']] ]
		if (np.max(crop_coords) > 2*rad or np.min(crop_coords) < 0):
			if (verb > 2):
				import pylab as plt
				from matplotlib.collections import PatchCollection
				show_shwfs_vecs(subaps[:,::2]*0, subaps, img=None, extent=extent, title=None, scale=10, pause=False, fignum=None, keep=True)
				aprad = plt.Circle(tuple(zerncntr[::-1]), radius=rad, alpha=0.5)
				thisgca = plt.gca(); thisgca.add_artist(aprad)
				raw_input("...")
			raise ValueError("Not all sub apertures in Zernike radius!")

	slopes = (np.indices(sasize, dtype=float)/(np.r_[sasize].reshape(-1,1,1))).reshape(2,-1)
	slopes2 = np.vstack([slopes, slopes[0]*0+1])
	slopesi = np.linalg.pinv(slopes2)

	zernslopes = np.r_[ [[calc_slope(zbase[subap[0]+saoffs[0]:subap[1]+saoffs[0], subap[2]+saoffs[1]:subap[3]+saoffs[1]], slopesi=slopesi) for subap in subaps] for zbase in zbasis['modes']] ].reshape(nzern, -1)


	if (verb>2):
		# Inspect Zernike influence matrix
		import pylab as plt
		from matplotlib.collections import PatchCollection
		for zidx, (zslope, zbase) in enumerate(zip(zernslopes, zbasis['modes'])):
			plslope = (zslope/np.abs(zslope).mean()).reshape(-1,2)
			refpos = (subaps[:,1::2] + subaps[:,::2])/2

			show_shwfs_vecs(plslope, subaps, img=zbase, extent=extent, title=None, scale=10, pause=False, fignum=None, keep=True)
			plt.plot(zerncntr[1], zerncntr[0], 'p', markersize=20)

			raw_input("continue...")
			plt.close()

	# np.linalg.pinv() takes the cutoff wrt the *maximum*, we want a cut-off
	# based on the cumulative sum, i.e. the total included power, which is 
	# why we use svd() and not pinv().
	U, s, Vh = np.linalg.svd(zernslopes*sfac, full_matrices=False)
	cums = s.cumsum()/s.sum()
	nvec = np.argwhere(cums >= singval)[0][0]
	singval = cums[nvec]
	s[nvec+1:] = np.inf
	zern_inv_mat = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

	if (check):
		pseudo_i = np.dot(zern_inv_mat, zernslopes)
		quality = np.trace(pseudo_i)/np.sum(pseudo_i)
		if (verb>2):
			print "calc_zern_infmat(): quality: %.g, singval: %.g"
		# @todo What was the idea of this check?
		#if (quality < singval*0.8):

	return zern_inv_mat, zernslopes*sfac, zbasis, extent


def find_mla_grid(wfsimg, size, clipsize=None, minif=0.6, nmax=-1, copy=True, method='boundsclip', sort=False, verb=0):
	"""
	Given a Shack-hartmann wave front sensor image, find a grid of 
	subapertures (sa) of approximately **size** big.

	The image will be destroyed during analysis, unless **copy** is True 
	(default).

	The returned MLA grid coordinates can be as center coordinates or as 
	bounds, formatted as (min0, max0, min1, max1) for each subaperture. If 
	the method is 'boundsclip', these bounds are additionally clipped to
	the input image size such that they are safe to be used as slicing 
	bounds.
	
	@param [in] wfsimg Image to analyze
	@param [in] size Subaperture size [pixels]
	@param [in] size Size to clip out when searching for spots [pixels]
	@param [in] minif Minimimum pixel value to have to consider it as a new subaperture, as fraction of the global maximum.
	@param [in] nmax Maximum number of subapertures to search for (-1 for no max)
	@param [in] copy Copy image before modifying
	@param [in] method Coordinate format to return, either 'bounds', 'boundsclip' or center'
	@param [in] sort Sort subaperture coordinates
	@param [in] verb Plot intermediate results
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
		if (method == 'boundsclip'):
			newsa = (max(p[0] - size[0]/2, 0), 
					min(p[0] + size[0]/2, wfsimg.shape[0]),
					max(p[1] - size[1]/2, 0),
					min(p[1] + size[1]/2, wfsimg.shape[1]))
		elif (method == 'bounds'):
			newsa = (p[0] - size[0]/2, 
					p[0] + size[0]/2,
					p[1] - size[1]/2,
					p[1] + size[1]/2)
		else:
			newsa = tuple(p)
		subap_grid.append(newsa)
		
		# Clear out this subaperture so we don't add it again. To make sure 
		# we don't clip the whole image, clip the indices
		wfsimg[max(p[0]-cs[0]/2,0):min(p[0]+cs[0]/2,wfsimg.shape[0]), max(p[1]-cs[1]/2,0):min(p[1]+cs[1]/2,wfsimg.shape[1])] = wfsimg.min()
		
		if (nmax > 0 and len(subap_grid) >= nmax):
			break

	subap_grid = np.r_[subap_grid]

	if (sort):
		# Sort by increasing pixel
		sortidx = np.argsort( (subap_grid[:,0]/size[0]).astype(int)*wfsimg.shape[0] + subap_grid[:,2])
	else:
		sortidx = slice(None)

	if (verb > 2):
		import pylab as plt
		from matplotlib.collections import PatchCollection
		# MLA grid as rectangles for plotting
		mlapatches_im = [ plt.Rectangle((subap[1], subap[0]), size[0], size[1], fc='none', ec='k') for subap in subap_grid[:,::2] ]

		# Inspect MLA grid parsing
		plt.figure();
		plt.imshow(wfsimg)
		plt.title("MLA grid")
		thisgca = plt.gca()
		thisgca.add_collection(PatchCollection(mlapatches_im, match_original=True))
		raw_input("...")
		plt.close()

	
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
			plt.title("actid=%d, meaninf=%g" % (actid, meaninf))
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

def show_shwfs_vecs(shiftvec, subaps, refpos=None, img=None, extent=None, title=None, scale=10, pause=False, fignum=None, keep=True):
	"""
	Show a SHWFS measurements with subapertures and potentially a background 
	image. If **refpos** is given, SHWFS shift vectors will be plotted at 
	those locations, otherwise at the center of the sub apertures.

	Additionally, a background image (wavefront, sensor image) can be 
	plotted along with the measurements in the background. If the coordinate 
	system is different than that of the sub apertures, this can be fixed by 
	using the **extent** keyword.

	**Scale** tweaks the arrow length, unity being the approximate real size.
	The default is 10 to enhance the visibility.

	@param [in] shiftvec (N,2) vector with shifts
	@param [in] subaps (N,4) vector with sub aperture positions as (low0, high0, low1, high1)
	@param [in] refpos Positions to plot shift vectors at (if None use subaps center)
	@param [in] img Background image, e.g. a wavefront
	@param [in] extent Extent for the image for imshow()
	@param [in] title Plot title
	@param [in] scale Scale used for arrow width, unity is approximately
	@param [in] pause Pause before continuing
	@param [in] fignum Figure number to use for figure()
	@param [in] keep Keep plot window open after returning
	"""

	import pylab as plt
	from matplotlib.collections import PatchCollection

	sasize = np.r_[subaps[:,:2].ptp(1).mean(), subaps[:,2:].ptp(1).mean()]
	subaps_im = [ plt.Rectangle((subap[1], subap[0]), sasize[0], sasize[1], fc='none', ec='k') for subap in subaps[:,::2] ]


	plt.figure(fignum); plt.clf()
	plt.title(title or "SHWFS vectors on MLA grid")

	if (img != None): plt.imshow(img, extent=extent)

	if (refpos == None): refpos = subaps[:,::2]+sasize/2

	q = plt.quiver(refpos[:,1], refpos[:,0], shiftvec[:,1], shiftvec[:,0], angles='xy', scale=refpos.ptp()/10., color='r')
	p = plt.quiverkey(q, refpos.min(0)[1], refpos.min(0)[0], 5, "5 pix.", coordinates='data', color='r')
	
	thisgca = plt.gca()
	thisgca.add_collection(PatchCollection(subaps_im, match_original=True))

	if (pause): raw_input("Press any key to continue...")
	if (not keep): plt.close()

def sim_shwfs(wave, mlagrid, pad=True, scale=2):
	"""
	Simulate the SHWFS image of a complex wave, using **mlagrid** for 
	lenslet positions. The complex wave is defined as:

		E \propto A \exp(-i*phi)

	i.e.

		e_wave = amp * np.exp(-1j*phi)

	@returns Float array of equal shape as **wave** with the SHWFS image
	"""

	assert wave.dtype in [np.complex64, np.complex128, np.complex256, complex], "sim_shwfs(): require complex wave as input"

	sasz = (mlagrid[0][1::2] - mlagrid[0][::2]).astype(int)
	assert sasz[0] == sasz[1], "non-square subapertures not supported"
	sasz = sasz[0]

	shwfs = np.zeros(wave.shape, dtype=wave.dtype)

	# Slow code, left here as illustration
	# for mla in mlagrid:
	# 	# Crop subaperture
	# 	wavecrop = wave[mla[0]:mla[1], mla[2]:mla[3]]
	# 	# Pad with zeros
	# 	if (pad):
	# 		wavecrop = tim.fft.embed_data(wavecrop, scale=scale)
	# 	# FFT, compute power, shift to irigin
	# 	wavecropft = np.fft.fftshift(np.abs(np.fft.fft2(wavecrop))**2.)
	# 	# De-pad, insert in larger image
	# 	if (pad):
	# 		wavecropft = tim.fft.embed_data(wavecropft, -1, scale=scale)
	# 	shwfs[mla[0]:mla[1], mla[2]:mla[3]] = wavecropft

	# Faster code, does the same as above except skipping some data shuffling
	try:
		fftfunc = pyfftw.interfaces.numpy_fft.fft2
		wavecrop = pyfftw.n_byte_align_empty((sasz, sasz), 16, 'complex128')
		pyfftw.interfaces.cache.enable()
	except:
		fftfunc = np.fft.fft2
		wavecrop = np.zeros((sasz, sasz), dtype=wave.dtype)

	for mla in mlagrid:
		# Crop subaperture (2.5µs)
		wavecrop[:] = wave[mla[0]:mla[1], mla[2]:mla[3]]

		# FFT (150µs)
		wavecropft = fftfunc(wavecrop, s = wavecrop.shape*np.r_[scale])

		# Replace into image array (4 * 7.5µs), quadrant by quadrant
		shwfs[mla[0]:mla[1]-sasz/2, mla[2]:mla[3]-sasz/2] = wavecropft[-sasz/2:, -sasz/2:]
		shwfs[mla[0]+sasz/2:mla[1], mla[2]:mla[3]-sasz/2] = wavecropft[:sasz/2, -sasz/2:]
		shwfs[mla[0]:mla[1]-sasz/2, mla[2]+sasz/2:mla[3]] = wavecropft[-sasz/2:, :sasz/2]
		shwfs[mla[0]+sasz/2:mla[1], mla[2]+sasz/2:mla[3]] = wavecropft[:sasz/2, :sasz/2]
	
	# Compute power (550µs)
	return np.abs(shwfs)**2.0

if __name__ == "__main__":
	sys.exit(unittest.main())
