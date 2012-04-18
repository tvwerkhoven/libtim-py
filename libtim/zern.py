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

#=============================================================================
# Import libraries here
#=============================================================================

import numpy as N
import unittest

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

from scipy.misc import factorial as fac
def zernike_rad(m, n, rho):
	"""
	Make radial Zernike polynomial on coordinate grid **rho**.

	@param [in] m Radial Zernike index
	@param [in] n Azimuthal Zernike index
	@param [in] rho Radial coordinate grid
	@return Radial polynomial with identical shape as **rho**
	"""
	if (N.mod(n-m, 2) == 1):
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
	if (m > 0): return nc*zernike_rad(m, n, rho) * N.cos(m * phi)
	if (m < 0): return nc*zernike_rad(-m, n, rho) * N.sin(-m * phi)
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
# 	grid = (N.indices((size, size), dtype=N.float) - 0.5*size) / (0.5*size)
# 	grid_rad = (grid[0]**2. + grid[1]**2.)**0.5
# 	grid_ang = N.arctan2(grid[0], grid[1])
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
	return N.asanyarray(norms)

### Higher level Zernike generating / fitting functions

def calc_zern_basis(nmodes, rad, mask=True):
	"""
	Calculate a basis of **nmodes** Zernike modes with radius **rad**.

	If **mask** is true, set everything outside of radius **rad** to zero (default). If this is not done, the set of Zernikes will be **rad** by **rad** square and are not orthogonal anymore.

	This output of this function can be used as cache for other functions.

	@param [in] nmodes Number of modes to generate
	@param [in] rad Radius of Zernike modes
	@param [in] mask Mask area outside Zernike modes or not
	@return Dict with entries 'basis' a list of Zernike modes and 'covmat' a covariance matrix for all these modes with 'covmat_in' its inverse.
	"""

	if (rad <= 0):
		raise ValueError("radius should be > 0")

	if (nmodes <= 0):
		return {'modes':[], 'covmat':0, 'covmat_in':0}

	# Use vectors instead of a grid matrix
	rvec = ((N.arange(2.0*rad) - rad)/rad)
	r0 = rvec.reshape(-1,1)
	r1 = rvec.reshape(1,-1)
	grid_rad = (r1**2. + r0**2.)**0.5
	grid_ang = N.arctan2(r0, r1)

	if (mask):
		grid_mask = grid_rad <= 1
	else:
		grid_mask = 1

	# Build list of Zernike modes
	zern_modes = [zernikel(zmode+1, grid_rad, grid_ang) * grid_mask for zmode in xrange(nmodes)]

	# Calculate covariance matrix
	cov_mat = N.array([[N.sum(zerni * zernj) for zerni in zern_modes] for zernj in zern_modes])
	# Invert covariance matrix using SVD
	cov_mat_in = N.linalg.pinv(cov_mat)

	# Create and return dict
	return {'modes': zern_modes, 'covmat':cov_mat, 'covmat_in':cov_mat_in}

def fit_zernike(wavefront, zern_data={}, nmodes=10, startmode=1, fitweight=None, center=(-0.5, -0.5), rad=-0.5, err=None):
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
		center = -N.r_[center] * min(wavefront.shape)

	# Use vectors instead of grid
	rvec = ((N.arange(2*rad) - rad)/rad)
	r0 = rvec.reshape(-1,1)
	r1 = rvec.reshape(1,-1)
	grid_rad = (r0**2. + r1**2.)**0.5
	grid_mask = grid_rad <= 1

	xslice = slice(center[0]-rad, center[0]+rad)
	yslice = slice(center[1]-rad, center[1]+rad)

	# Compute Zernike basis if absent
	if (not zern_data.has_key('modes')):
		tmp_zern = calc_zern_basis(nmodes, rad)
		zern_data['modes'] = tmp_zern['modes']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
	# Compute Zernike basis if insufficient
	elif (nmodes > len(zern_data['modes']) or
		zern_data['modes'][0].shape != grid_mask.shape):
		tmp_zern = calc_zern_basis(nmodes, rad)
		# This data already exists, overwrite it with new data
		zern_data['modes'] = tmp_zern['modes']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
	zern_basis = zern_data['modes']

	zern_basis = zern_data['modes'][:nmodes]
	zern_covmat_in = zern_data['covmat_in'][:nmodes, :nmodes]
	# Calculate Zernike covariance matrix
# 	cov_mat = N.zeros((nmodes, nmodes))
# 	for modei in xrange(nmodes):
# 		zerni = zern_basis[modei]
# 		for modej in xrange(nmodes):
# 			zernj = zern_basis[modej]
# 			cov_mat[modei, modej] = N.sum(zerni * zernj)
	# This is the same as above
#	global GLOB_ZERN_COVMAT_IN # will be updated by calc_zern_basis()

	# Calculate inner products
	## @bug This weight is wrong, it modifies the data instead of weighing it in the fit.
	weight = grid_mask
	wf_zern_inprod = 0
	if (fitweight != None):
		raise RuntimeWarning("Warning: weighed fitting is broken, not using")
# 		weight = fitweight[yslice, xslice] * grid_mask
# 		# Normalize weight such that the mean is 1
# 		weight /= weight[grid_mask].mean()
# 		wf_zern_inprod = N.array([N.sum(wavefront[yslice, xslice] * zmode * weight) for zmode in zern_basis])
# 	else:
	wf_zern_inprod = N.array([N.sum(wavefront[yslice, xslice] * zmode) for zmode in zern_basis])

	# Calculate Zernike amplitudes
	wf_zern_vec = N.dot(zern_covmat_in, wf_zern_inprod)
	wf_zern_vec[:startmode-1] = 0

	# Calculate full Zernike phase
	wf_zern_rec = calc_zernike(wf_zern_vec, zern_data=zern_data, rad=min(wavefront.shape)/2)
	# Calculate errors
	fitdiff = wf_zern_rec - wavefront[yslice, xslice]
	# Make sure value outside mask is same as mean inside mask
	fitdiff[grid_mask == False] = fitdiff[grid_mask].mean()

	if (err != None):
		err.append((fitdiff**2.0).mean())
		err.append(N.abs(fitdiff).mean())
		err.append(N.abs(fitdiff).mean()**2.0)

	return (wf_zern_vec, wf_zern_rec, fitdiff)

def calc_zernike(zern_vec, rad, zern_data={}):
	"""
	Construct wavefront with Zernike amplitudes **zern_vec**.

	Given vector **zern_vec** with the amplitude of Zernike modes, return the reconstructed wavefront with radius **rad**.

	This function uses **zern_data** as cache. If this is not given, it will be generated. See calc_zern_basis() for details.

	@param [in] zern_vec 1D vector of Zernike amplitudes
	@param [in] rad Radius for Zernike modes to construct
	@param [in] zern_data Zernike basis cache
	@see See calc_zern_basis() for details on **zern_data** cache
	"""

	# Compute Zernike basis if absent
	if (not zern_data.has_key('modes')):
		tmp_zern = calc_zern_basis(len(zern_vec), rad)
		zern_data['modes'] = tmp_zern['modes']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
	# Compute Zernike basis if insufficient
	elif (len(zern_vec) > len(zern_data['modes'])):
		tmp_zern = calc_zern_basis(len(zern_vec), rad)
		# This data already exists, overwrite it with new data
		zern_data['modes'] = tmp_zern['modes']
		zern_data['covmat'] = tmp_zern['covmat']
		zern_data['covmat_in'] = tmp_zern['covmat_in']
	zern_basis = zern_data['modes']

	# Reconstruct the wavefront by summing modes
	return reduce(lambda x,y: x+y[1]*zern_basis[y[0]], enumerate(zern_vec), 0)

class TestZernikes(unittest.TestCase):
	def setUp(self):
		"""Generate source image, darkfield, flatfield and simulated data"""
		rad = 257
		rad = 127
		self.rad = rad
		self.nmodes = 25
		self.vec = N.random.random(self.nmodes)
		self.basis = []
		self.basis_data = calc_zern_basis(self.nmodes, self.rad)
		self.basis = self.basis_data['modes']
		self.wf = reduce(lambda x,y: x+y[1]*self.basis[y[0]], enumerate(self.vec), 0)

	# Shallow data tests
	def test0a_basis_data(self):
		"""Length of basis should be the same as amplitude vector length"""
		self.assertEqual(len(self.basis), len(self.vec))

	def test0b_data(self):
		"""Shape of basis and test wf should be equal"""
		for mode in self.basis:
			self.assertEqual(self.wf.shape, mode.shape)

	def test0c_indices(self):
		"""Test Noll to Zernike index conversion, see
		<https://oeis.org/A176988>"""
		zern = [(0,0), (1,1), (1,-1), (2,0), (2,-2), (2,2), (3,-1), (3,1), (3,-3), (3,3)]
		for j, nm in enumerate(zern):
			self.assertEqual(noll_to_zern(j+1), nm)

	# Shallow function test
	def test1a_zero_wf(self):
		"""Zero-wavefront should return zero amplitude vector"""
		fitdata = fit_zernike(N.zeros((64, 64)), nmodes=self.nmodes)
		fitvec = fitdata[0]
		self.assertTrue(N.allclose(fitvec, 0.0))

	def test1b_shape_detection(self):
		"""Test calc_zern_basis basis shape detection"""
		basis_data = calc_zern_basis(nmodes=10, rad=64)
		basis = basis_data['modes']
		self.assertEqual(len(basis), 10)
		self.assertEqual(basis[0].shape, (128, 128))
		self.assertEqual(basis[0].shape, basis[-1].shape)

		basis_data = calc_zern_basis(nmodes=5, rad=128)
		basis = basis_data['modes']
		self.assertEqual(len(basis), 5)
		self.assertEqual(basis[0].shape, (256, 256))
		self.assertEqual(basis[0].shape, basis[-1].shape)

		basis_data = calc_zern_basis(nmodes=5, rad=64)
		basis = basis_data['modes']
		self.assertEqual(len(basis), 5)
		self.assertEqual(basis[0].shape, (128, 128))
		self.assertEqual(basis[0].shape, basis[-1].shape)

	def test1c_file_write(self):
		"""Test disk writing."""
		pyfits.writeto('TestZernikes-modes.fits', N.r_[self.basis], clobber=True)

	# Deep function tests
	# calc_zern_basis(nmodes, rad, mask=True):
	# fit_zernike(wavefront, nmodes=10, center=(-0.5, -0.5), rad=-0.5):
	# calc_zernike(zern_vec, rad=-1):

	def test2b_zern_calc(self):
		"""Compare calc_zernike output with pre-computed basis"""
		vec = [0]*self.nmodes
		for i in xrange(self.nmodes):
			vec[i] = 1
			testzern = calc_zernike(vec, self.rad)
			self.assertTrue(N.allclose(self.basis[i], testzern))
			vec[i] = 0

	def test2c_variance(self):
		"""Test whether all Zernike modes have variance unity"""
		rad = self.rad
		grid = (N.indices((2*rad, 2*rad), dtype=N.float) - rad) / rad
		grid_rad = (grid[0]**2. + grid[1]**2.)**0.5
		self.mask = grid_rad <= 1

		for idx, m in enumerate(self.basis):
			if (idx == 0):
				continue
			# The error in the variance should scale with the number of
			# pixels, more pixels means less error because of better sampling.
			# Because of this we take 1/npixels as an error margin. The factor
			# 1.1 is added for numerical roundoff and other computer errors.
			# We use the mask to test only the relevant part of the Zernike
			# modes
			self.assertAlmostEqual(N.var(m[self.mask]), 1.0, delta=1.1/(self.rad**2.)**0.5)

	def test2d_equal_mode(self):
		"""Test equal-mode Zernike reconstruction"""
		fitdata = fit_zernike(self.wf, nmodes=self.nmodes)
		fitvec = fitdata[0]
		self.assertAlmostEqual(N.sum(self.vec - fitvec), 0.0)
		self.assertTrue(N.allclose(self.vec, fitvec))

	def test2e_unequal_mode(self):
		"""Test unequal-mode Zernike reconstruction"""
		fitdata = fit_zernike(self.wf, nmodes=10)
		fitvec = fitdata[0]
		self.assertAlmostEqual(N.mean(self.vec[:10] / fitvec), 1.0, delta=0.1)
		self.assertTrue(N.allclose(self.vec[:10]/fitvec, 1.0, rtol=0.1))

	def test2f_fit_startmode(self):
		"""Test startmode parameter in fit_zernike"""
		# startmode == 0 should raise an error, as this is not a valid Noll
		# index
		with self.assertRaises(ValueError):
			fitdata = fit_zernike(self.wf, nmodes=10, startmode=0)

		# Setting startmode higher should block out the first few modes
		for s in range(10):
			fitdata = fit_zernike(self.wf, nmodes=10, startmode=s+1)
			fitvec = fitdata[0]
			self.assertEqual(tuple(fitvec[:s]), (0,)*s)

class TestZernikeSpeed(unittest.TestCase):
	def setUp(self):
		self.calc_iter = 3
		self.fit_iter = 5
		self.nmodes = 25
		self.rad = 257

	def test3a_timing_calc(self):
		"""Test Zernike calculation timing and cache functioning"""

		t1 = Timer("""
a=calc_zernike(vec, rad, z_cache)
		""", """
from __main__ import calc_zern_basis, fit_zernike, calc_zernike
import numpy as N
rad = %d
nmodes = %d
vec = N.random.random(nmodes)
z_cache = {}
		""" % (self.rad, self.nmodes) )
		t2 = Timer("""
a=calc_zernike(vec, rad, {})
		""", """
from __main__ import calc_zern_basis, fit_zernike, calc_zernike
import numpy as N
rad = %d
nmodes = %d
vec = N.random.random(nmodes)
		""" % (self.rad, self.nmodes) )
		t_cache = t1.timeit(self.calc_iter)/self.calc_iter
		t_nocache = t2.timeit(self.calc_iter)/self.calc_iter
		# Caching should be at least twice as fast as no caching
		# Note that here we do not initialize the cache in the setup, it is
		# set to an empty dict which is filled on first run. This test should
		# test that this automatic filling works properly
		self.assertGreater(t_nocache/2.0, t_cache)

	def test3b_timing_calc(self):
		"""Test Zernike calculation performance with and without cache"""

		t1 = Timer("""
a=calc_zernike(vec, rad, z_cache)
		""", """
from __main__ import calc_zern_basis, fit_zernike, calc_zernike
import numpy as N
rad = %d
nmodes = %d
vec = N.random.random(nmodes)
z_cache = calc_zern_basis(len(vec), rad)
		""" % (self.rad, self.nmodes) )

		t2 = Timer("""
a=calc_zernike(vec, rad, {})
		""", """
from __main__ import calc_zern_basis, fit_zernike, calc_zernike
import numpy as N
rad = %d
nmodes = %d
vec = N.random.random(nmodes)
		""" % (self.rad, self.nmodes) )

		t_cached = min(t1.repeat(2, self.calc_iter))/self.calc_iter
		t_nocache = min(t2.repeat(2, self.calc_iter))/self.calc_iter
		print "test3b_timing_calc(): rad=257, nmodes=25 cache: %.3g s/it no cache: %.3g s/it" % (t_cached, t_nocache)

	def test3c_timing_fit(self):
		"""Test Zernike fitting performance"""

		t1 = Timer("""
a=fit_zernike(wf, z_cache, nmodes=nmodes)
		""", """
from __main__ import calc_zern_basis, fit_zernike, calc_zernike
import numpy as N
rad = %d
nmodes = %d
vec = N.random.random(nmodes)
z_cache = calc_zern_basis(len(vec), rad)
wf = N.random.random((rad, rad))
		""" % (self.rad, self.nmodes) )

		t_cached = min(t1.repeat(2, self.fit_iter))/self.fit_iter
		# Caching should be at least twice as fast as no caching
		print "test3c_timing_fit(): rad=257, nmodes=25 %.3g sec/it" % (t_cached)

if __name__ == "__main__":
	import sys
	import pyfits
	from timeit import Timer
	sys.exit(unittest.main())
