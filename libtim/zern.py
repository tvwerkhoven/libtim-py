#!/usr/bin/env python
# encoding: utf-8
"""
@file zern.py
@brief Zernike basis function utilities

@package libtim.zern
@brief Zernike basis function utilities
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120403

Construct and analyze Zernike basis functions
"""

#==========================================================================
# Import libraries here
#==========================================================================

import numpy as np
import libtim as tim
import libtim.im

#==========================================================================
# Defines
#==========================================================================

#==========================================================================
# Routines
#==========================================================================

from scipy.misc import factorial as fac
def zernike_rad(m, n, rho):
	"""
	Make radial Zernike polynomial on coordinate grid **rho**.

	@param [in] m Radial Zernike index
	@param [in] n Azimuthal Zernike index
	@param [in] rho Radial coordinate grid
	@return Radial polynomial with identical shape as **rho**
	"""
	if (np.mod(n-m, 2) == 1):
		return rho*0.0

	wf = rho*0.0
	for k in range((n-m)/2+1):
		wf += rho**(n-2.0*k) * (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )

	return wf

def zernike(m, n, rho, phi, norm=True):
	"""
	Calculate Zernike mode (m,n) on grid **rho** and **phi**.

	**rho** and **phi** should be radial and azimuthal coordinate grids of identical shape, respectively.

	@param [in] m Radial Zernike index
	@param [in] n Azimuthal Zernike index
	@param [in] rho Radial coordinate grid
	@param [in] phi Azimuthal coordinate grid
	@param [in] norm Normalize modes to unit variance
	@return Zernike mode (m,n) with identical shape as rho, phi
	@see <http://research.opt.indiana.edu/Library/VSIA/VSIA-2000_taskforce/TOPS4_2.html> and <http://research.opt.indiana.edu/Library/HVO/Handbook.html>.
	"""
	nc = 1.0
	if (norm):
		nc = (2*(n+1)/(1+(m==0)))**0.5
	if (m > 0): return nc*zernike_rad(m, n, rho) * np.cos(m * phi)
	if (m < 0): return nc*zernike_rad(-m, n, rho) * np.sin(-m * phi)
	return nc*zernike_rad(0, n, rho)

def noll_to_zern(j):
	"""
	Convert linear Noll index to tuple of Zernike indices.

	j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike index.

	@param [in] j Zernike mode Noll index
	@return (n, m) tuple of Zernike indices
	@see <https://oeis.org/A176988>.
	"""
	if (j == 0):
		raise ValueError("Noll indices start at 1, 0 is invalid.")

	n = 0
	j1 = j-1
	while (j1 > n):
		n += 1
		j1 -= n

	m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
	return (n, m)

def zernikel(j, rho, phi, norm=True):
	n, m = noll_to_zern(j)
	return zernike(m, n, rho, phi, norm)

# def zernikel(j, size=256, norm=True):
# 	"""
# 	Calculate Zernike mode with Noll-index j on a square grid of <size>^2
# 	elements
# 	"""
# 	n, m = noll_to_zern(j)
#
# 	grid = (np.indices((size, size), dtype=np.float) - 0.5*size) / (0.5*size)
# 	grid_rad = (grid[0]**2. + grid[1]**2.)**0.5
# 	grid_ang = np.arctan2(grid[0], grid[1])
# 	return zernike(m, n, grid_rad, grid_ang, norm)

def noll_to_zern_broken(j):
	"""
	Previous and incorrect Noll-to-Zernike conversion.

	Stored for reference purposes. Fixed around 1321970330.98158 or Tue Nov 22 13:59:10 2011 UTC. All data generated before this has invalid mapping.
	@deprecated Incorrect mapping, use noll_to_zern instead
	"""
	raise DeprecatedWarning("Incorrect mapping, use noll_to_zern instead")
	n = 0
	j1 = j
	while (j1 > n):
		n += 1
		j1 -= n
	m = -n+2*j1
	return (n, m)

def fix_noll_map(max):
	"""
	Translate old incorrect Noll coordinates to correct values.

	This function repairs data generated with noll_to_zern_broken().
	"""
	return [(jold, jnew)
		for jold in xrange(max)
			for jnew in xrange(1, max)
				if (noll_to_zern_broken(jold) == noll_to_zern(jnew))]

def zern_normalisation(nmodes=30):
	"""
	Calculate normalisation vector.

	This function calculates a **nmodes** element vector with normalisation constants for Zernike modes that have not already been normalised.

	@param [in] nmodes Size of normalisation vector.
	@see <http://research.opt.indiana.edu/Library/VSIA/VSIA-2000_taskforce/TOPS4_2.html> and <http://research.opt.indiana.edu/Library/HVO/Handbook.html>.
	"""

	nolls = (noll_to_zern(j+1) for j in xrange(nmodes))
	norms = [(2*(n+1)/(1+(m==0)))**0.5 for n, m  in nolls]
	return np.asanyarray(norms)

### Higher level Zernike generating / fitting functions

def calc_zern_basis(nmodes, rad, modestart=1, calc_covmat=False):
	"""
	Calculate a basis of **nmodes** Zernike modes with radius **rad**.

	((If **mask** is true, set everything outside of radius **rad** to zero (default). If this is not done, the set of Zernikes will be **rad** by **rad** square and are not orthogonal anymore.)) --> Nothing is masked, do this manually using the 'mask' entry in the returned dict.

	This output of this function can be used as cache for other functions.

	@param [in] nmodes Number of modes to generate
	@param [in] rad Radius of Zernike modes
	@param [in] modestart First mode to calculate (Noll index, i.e. 1=piston)
	@param [in] calc_covmat Return covariance matrix for Zernike modes, and its inverse
	@return Dict with entries 'modes' a list of Zernike modes, 'modesmat' a matrix of (nmodes, npixels), 'covmat' a covariance matrix for all these modes with 'covmat_in' its inverse, 'mask' is a binary mask to crop only the orthogonal part of the modes.
	"""

	if (nmodes <= 0):
		return {'modes':[], 'modesmat':[], 'covmat':0, 'covmat_in':0, 'mask':[[0]]}
	if (rad <= 0):
		raise ValueError("radius should be > 0")
	if (modestart <= 0):
		raise ValueError("**modestart** Noll index should be > 0")

	# Use vectors instead of a grid matrix
	rvec = ((np.arange(2.0*rad) - rad)/rad)
	r0 = rvec.reshape(-1,1)
	r1 = rvec.reshape(1,-1)
	grid_rad = tim.im.mk_rad_mask(2*rad)
	grid_ang = np.arctan2(r0, r1)

	grid_mask = grid_rad <= 1

	# Build list of Zernike modes, these are *not* masked/cropped
	zern_modes = [zernikel(zmode, grid_rad, grid_ang) for zmode in xrange(modestart, nmodes+modestart)]

	# Convert modes to (nmodes, npixels) matrix
	zern_modes_mat = np.r_[zern_modes].reshape(nmodes, -1)

	covmat = covmat_in = None
	if (calc_covmat):
		# Calculate covariance matrix
		covmat = np.array([[np.sum(zerni * zernj * grid_mask) for zerni in zern_modes] for zernj in zern_modes])
		# Invert covariance matrix using SVD
		covmat_in = np.linalg.pinv(covmat)

	# Create and return dict
	return {'modes': zern_modes, 'modesmat': zern_modes_mat, 'covmat':covmat, 'covmat_in':covmat_in, 'mask': grid_mask}

def fit_zernike(wavefront, zern_data={}, nmodes=10, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5, rec_zern=True, err=None):
	"""
	Fit **nmodes** Zernike modes to a **wavefront**.

	The **wavefront** will be fit to Zernike modes for a circle with radius **rad** with origin at **center**. **weigh** is a weighting mask used when fitting the modes.

	If **center** or **rad** are between 0 and -1, the values will be interpreted as fractions of the image shape.

	**startmode** indicates the Zernike mode (Noll index) to start fitting with, i.e. ***startmode**=4 will skip piston, tip and tilt modes. Modes below this one will be set to zero, which means that if **startmode** == **nmodes**, the returned vector will be all zeroes. This parameter is intended to ignore low order modes when fitting (piston, tip, tilt) as these can sometimes not be derived from data.

	If **err** is an empty list, it will be filled with measures for the fitting error:
	1. Mean squared difference
	2. Mean absolute difference
	3. Mean absolute difference squared

	This function uses **zern_data** as cache. If this is not given, it will be generated. See calc_zern_basis() for details.

	@param [in] wavefront Input wavefront to fit
	@param [in] zern_data Zernike basis cache
	@param [in] nmodes Number of modes to fit
	@param [in] startmode Start fitting at this mode (Noll index)
	@param [in] fitweight Mask to use as weights when fitting
	@param [in] center Center of Zernike modes to fit
	@param [in] rad Radius of Zernike modes to fit
	@param [in] rec_zern Reconstruct Zernike modes and calculate errors.
	@param [out] err Fitting errors
	@return Tuple of (wf_zern_vec, wf_zern_rec, fitdiff) where the first element is a vector of Zernike mode amplitudes, the second element is a full 2D Zernike reconstruction and the last element is the 2D difference between the input wavefront and the full reconstruction.
	@see See calc_zern_basis() for details on **zern_data** cache
	"""

	if (rad < -1 or min(center) < -1):
		raise ValueError("illegal radius or center < -1")
	elif (rad > 0.5*max(wavefront.shape)):
		raise ValueError("radius exceeds wavefront shape?")
	elif (max(center) > max(wavefront.shape)-rad):
		raise ValueError("fitmask shape exceeds wavefront shape?")
	elif (startmode	< 1):
		raise ValueError("startmode<1 is not a valid Noll index")

	# Convert rad and center if coordinates are fractional
	if (rad < 0):
		rad = -rad * min(wavefront.shape)
	if (min(center) < 0):
		center = -np.r_[center] * min(wavefront.shape)

	# Make cropping slices to select only central part of the wavefront
	xslice = slice(center[0]-rad, center[0]+rad)
	yslice = slice(center[1]-rad, center[1]+rad)

	# Compute Zernike basis if absent
	if (not zern_data.has_key('modes')):
		tmp_zern = calc_zern_basis(nmodes, rad)
		zern_data['modes'] = tmp_zern['modes']
		zern_data['modesmat'] = tmp_zern['modesmat']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
		zern_data['mask'] = tmp_zern['mask']
	# Compute Zernike basis if insufficient
	elif (nmodes > len(zern_data['modes']) or
		zern_data['modes'][0].shape != (2*rad, 2*rad)):
		tmp_zern = calc_zern_basis(nmodes, rad)
		# This data already exists, overwrite it with new data
		zern_data['modes'] = tmp_zern['modes']
		zern_data['modesmat'] = tmp_zern['modesmat']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
		zern_data['mask'] = tmp_zern['mask']

	zern_basis = zern_data['modes'][:nmodes]
	zern_basismat = zern_data['modesmat'][:nmodes]
	grid_mask = zern_data['mask']

	wf_zern_vec = 0
	grid_vec = grid_mask.reshape(-1)
	if (fitweight != None):
		# Weighed LSQ fit with data. Only fit inside grid_mask

		# Multiply weight with binary mask, reshape to vector
		weight = ((fitweight[yslice, xslice])[grid_mask]).reshape(1,-1)

		# LSQ fit with weighed data
		wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1) * weight
		wf_zern_vec = np.dot(wf_w, np.linalg.pinv(zern_basismat[:, grid_vec] * weight)).ravel()
	else:
		# LSQ fit with data. Only fit inside grid_mask

		# Crop out central region of wavefront, then only select the orthogonal part of the Zernike modes (grid_mask)
 		wf_w = ((wavefront[yslice, xslice])[grid_mask]).reshape(1,-1)
		wf_zern_vec = np.dot(wf_w, np.linalg.pinv(zern_basismat[:, grid_vec])).ravel()

	wf_zern_vec[:startmode-1] = 0

	# Calculate full Zernike phase & fitting error
	if (rec_zern):
		wf_zern_rec = calc_zernike(wf_zern_vec, zern_data=zern_data, rad=min(wavefront.shape)/2)
		fitdiff = (wf_zern_rec - wavefront[yslice, xslice])
		fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()
	else:
		wf_zern_rec = None
		fitdiff = None

	if (err != None):
		# For calculating scalar fitting qualities, only use the area inside the mask
		fitresid = fitdiff[grid_mask == True]
		err.append((fitresid**2.0).mean())
		err.append(np.abs(fitresid).mean())
		err.append(np.abs(fitresid).mean()**2.0)

	return (wf_zern_vec, wf_zern_rec, fitdiff)

def calc_zernike(zern_vec, rad, zern_data={}, mask=True):
	"""
	Construct wavefront with Zernike amplitudes **zern_vec**.

	Given vector **zern_vec** with the amplitude of Zernike modes, return the reconstructed wavefront with radius **rad**.

	This function uses **zern_data** as cache. If this is not given, it will be generated. See calc_zern_basis() for details.

	If **mask** is True, set everything outside radius **rad** to zero, this is the default and will use orthogonal Zernikes. If this is False, the modes will not be cropped.

	@param [in] zern_vec 1D vector of Zernike amplitudes
	@param [in] rad Radius for Zernike modes to construct
	@param [in] zern_data Zernike basis cache
	@param [in] mask If True, set everything outside the Zernike aperture to zero, otherwise leave as is.
	@see See calc_zern_basis() for details on **zern_data** cache and **mask**
	"""

	# Compute Zernike basis if absent
	if (not zern_data.has_key('modes')):
		tmp_zern = calc_zern_basis(len(zern_vec), rad)
		zern_data['modes'] = tmp_zern['modes']
		zern_data['modesmat'] = tmp_zern['modesmat']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
		zern_data['mask'] = tmp_zern['mask']
	# Compute Zernike basis if insufficient
	elif (len(zern_vec) > len(zern_data['modes'])):
		tmp_zern = calc_zern_basis(len(zern_vec), rad)
		# This data already exists, overwrite it with new data
		zern_data['modes'] = tmp_zern['modes']
		zern_data['modesmat'] = tmp_zern['modesmat']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
		zern_data['mask'] = tmp_zern['mask']
	zern_basis = zern_data['modes']

	gridmask = 1
	if (mask):
		gridmask = zern_data['mask']

	# Reconstruct the wavefront by summing modes
	return reduce(lambda x,y: x+y[1]*zern_basis[y[0]] * gridmask, enumerate(zern_vec), 0)

