#!/usr/bin/env python
# encoding: utf-8
"""
@file lightcurve.py
@brief Process lightcurve data

@package libtim.lightcurve
@brief Process lightcurve data
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20121025

General light curve data functions, e.g. from Kepler or SWASP.
"""

#=============================================================================
# Import libraries here
#=============================================================================

import numpy as np
import scipy as sp
import scipy.ndimage.filters
import unittest
import libtim as tim
import pylab as plt
import math

#=============================================================================
# Defines
#=============================================================================

# Plot for user, wait to continue
PLOT_INTERACTIVE = 0x040

#=============================================================================
# UTILITIES
#=============================================================================

def calc_ph(time, period, time0=0):
	"""
	Calculate phase from time coordinate, period and time0
	"""
	return ((time - (time0-period/2.)) % period) / period -0.5

#=============================================================================
# DECORRELATION FIT
#=============================================================================

def make_cotrend_vecs(refvecs):
	"""
	Light curves on different CCDS look different. This is due to CCD 
	artefacts rather than the star itself. We use the Kepler De-correlation
	algorithm to remove instrument fluctuations from the data. 

	The first step is to make the cotrending vectors, which is done here.
	
	From <http://keplergo.arc.nasa.gov/ContributedSoftwareKepcotrend.shtml>:

	1) The time series photometry of each star on a specific detector channel is normalized by its own median flux.
	2) One (unity) is subtracted from each time series so that the median value of the light curve is zero.
	3) The time series is divided by the root-mean square of the photometry.
	4) The correlation between each time series on the CCD channel is calculated using the median and root-mean square normalized flux.
	5) The median absolute correlation is then calculated for each star.
	6) All stars on the channel are sorted into ascending order of correlation.
	7) The 50 percent most correlated stars are selected.
	8) The median normalized fluxes only (as opposed to the root-mean square normalized fluxes) are now used for the rest of the process
	9) Singular Value Decomposition is applied to the matrix of correlated sources to create orthonormal basis vectors from the U matrix [V.T in this case], sorted by their singular values.
	10) The archived cotrending basis vectors are a reduced-rank representation of the full set of basis vectors and consist of the 16 leading columns.

	This routine returns V.T and s from np.linalg.svd()

	@param [in] refvecs Iterable of reference data to make cotrend basis for
	@return Tuple of (decorrelation matrix, singular value vector)
	"""

	# 1: normalize by median
	refvecs_n1 = [ref/np.median(ref) for ref in refvecs]

	# 2, 3: subtract unity, normalize by RMS (RMS == sqrt(mean**2 + stddev**2))
	refvecs_n2 = [(ref-1.0)/(ref.mean()**2.0 + ref.var())**0.5 for ref in refvecs_n1]

	# 4: calculate correlation
	#refcorr1 = [[sum(r1*r2) for r1 in refvecs_n1] for r2 in refvecs_n1]
	refcorr2 = [[sum(r1*r2) for r1 in refvecs_n2] for r2 in refvecs_n2]

	# 5: median absolute correlation per star
	refcorr_ma = [np.median(np.abs(corr)) for corr in refcorr2]

	# 6, 7: sort by correlation, select most correlated stars
	reford = np.argsort(refcorr_ma)

	# 8: Select median normalized fluxes (select all)
	refvecs_subset = np.r_[refvecs_n1][reford]

	# 9: compute SVD of matrix of previously selected subset. V contains 
	# the ordered set of cotrending vectors to work with
	U, s, V = np.linalg.svd(refvecs_subset, full_matrices=False)

	return V, s

def cotrend_fit(indata, cotrend_mat, findoffset=True, mask=slice(None)):
	"""
	De-correlate input data with linear de-correlation vectors. Returns a 
	decorrelation fit. This basically least-square fits the basis vectors
	in **cotrend_mat** to **indata**. Shapes  of input data should match.

	The mask supplied will be used to least-squares fit the data only to that
	portion, but the returned fit is always the same shape as **indata**.
	
	See <http://keplergo.arc.nasa.gov/ContributedSoftwareKepcotrend.shtml> for details.
	
	@param [in] indata The data to de-correlate 
	@param [in] cotrend_mat Matrix containing de-correlation vectors
	@param [in] mask Datamask to use (e.g. phase != in transit)
	@return Tuple of (de-correlation fit, fit coefficients)
	"""
	
	# TvW: add array of ones to find offset
	if (findoffset):
		cotrend_mat = np.vstack([np.ones(cotrend_mat.shape[1]), cotrend_mat])

	# This is [U^{T} * U]^{-1} * U^{T}, i.e. the inverse of the cotrend matrix, using the inputmask
	# from http://keplergo.arc.nasa.gov/ContributedSoftwareKepcotrend.shtml
	cotrend_mat_i = np.linalg.pinv(cotrend_mat[:, mask])
	#cotrend_mat_i = np.dot(np.linalg.inv(np.dot(cotrend_mat[mask].T, cotrend_mat[mask])), cotrend_mat[mask].T)
	
	# Calculate coefficients with masked data matrix
	decorr_coeff = np.dot(cotrend_mat_i.T, indata[mask].reshape(-1,1))
	
	# But calculate the offset for all data matrix (we also want to offset the 
	# masked region)
	decorr_fit = np.dot(cotrend_mat.T, decorr_coeff).reshape(-1)
		
	return decorr_fit
	
def stellar_detrend(mflux, mtime, fluxerr, phase, mask=slice(None), opoly=2, ospline=3, tjump_thresh=0.05, fjump_thresh=0.005, plot=0, verb=0):
	"""
	De-trend data from (stellar) variability.
	- Break up in small chunks: if ∆t is big or ∆flux is big
	- For each chunk, fit data: if ndat <10: median; if ndat <200: O(opoly) poly; else: spline O(ospline)
	- De-trend data with fit
	
	@param [in] mflux Input flux
	@param [in] mtime Time corresponding to flux
	@param [in] fluxerr Error in flux measurements
	@param [in] fjump_thresh Flux jump threshold (fraction)
	@param [in] tjump_thresh Time jump threshold (absolute)
	@param [in] phase Phase of transit (for masking)
	@param [in] opoly Polynomial order to fit
	@param [in] ospline Spline order to fit
	@param [in] plot Plot verbosity
	@return Fit to data
	"""
	
	# Convert boolean mask to indices
	mask = (phase < -0.1) + (phase > 0.15)
	maskidx = np.argwhere(mask).ravel()
	
	# Calculate derivative for flux / time, flux is in percent
	fjump = (mflux[1:] - mflux[:-1]) / mflux[:-1]
	tjump = mtime[1:] - mtime[:-1]
	# Element N gives diff (N+1) - N, e.g. element 0 gives change 0->1.
	# We prepend a 0 (no jump) such that element such that for a jump at
	# X, we can take flux[:x] and flux[x:] and separate around the jump
	fjump = np.hstack([[0], fjump])
	tjump = np.hstack([[0], tjump])
	
	# Plot all flux jumps
	if (plot & PLOT_INTERACTIVE):
		plt.figure(20)
		plt.clf()
		plt.title("Flux jumps")
		plt.plot(mtime, mflux-mflux.mean())
		plt.plot(mtime, fjump)
		plt.ylabel("Flux")

	# Find flux jumps larger than fjump_thresh
	fjumpidx = np.abs(fjump) > fjump_thresh
	
	# Find time difference more than 0.05 days = 75 min (exp. time ~ 30 min)
	tjumpidx = (np.abs(tjump) > tjump_thresh)
	
	# Check if flux jumps are real:
	# For each flux jump:
	# - calculate the 20-point median before the jump
	# - calculate the 20-point median after the jump
	# - check if they differ by more than 3.5 the minimum stddev
	split = lambda flux, idx: flux[max(idx-19,0):idx], flux[idx:idx+20]
	
	nojump = []
	for thisjump in np.argwhere(fjumpidx).ravel():
		flux_1b, flux_2b = split(mflux, thisjump)
		flux_1 = mflux[max(thisjump-19,0):thisjump]
		m1, s1 = np.mean(flux_1), flux_1.std()
		flux_2 = mflux[thisjump:thisjump+20]
		assert flux_1b == flux_1
		assert flux_2b == flux_2
		
		m2, s2 = np.mean(flux_2), flux_2.std()
		if (np.abs(m2-m1) < 3.5*np.min([s1, s2])):
			nojump.append(thisjump)

	if (verb >= VERB_DEBG):
		print "Found %d/%d flux jumps are significant" % (sum(fjumpidx)-len(nojump), sum(fjumpidx))
	fjumpidx[nojump] = False

	# Inspect flux jumps manually
	if (False and plot & PLOT_INTERACTIVE):
		for jump in np.argwhere(fjumpidx):
			print "Flux jump @ %d (showing %d--%d)" % (jump, max(jump-25, 0), jump+25)
			if (jump in np.argwhere(tjumpidx)):
				print "Skipping flux jump, also time jump"
				continue
			plt.figure(40)
			plt.clf()
			fdat_left = mflux[max(jump-25, 0):jump]
			fdat_right = mflux[jump:jump+25]
			ftime_left = mtime[max(jump-25, 0):jump]
			#np.arange(len(fdat_left)) - len(fdat_left)
			ftime_right = mtime[jump:jump+25]
			#np.arange(len(fdat_right))
			jump_size = (np.mean(fdat_left) - np.mean(fdat_right))
			jump_size /= np.min([fdat_left.std(), fdat_right.std()])
			plt.title(GLOB_ARGS.tgtname + " flux jump @ %d = %g" % (jump, jump_size))
			plt.plot(ftime_left, fdat_left)
			plt.plot(ftime_right, fdat_right)
			plt.ylabel("Flux")
			plt.xlabel("Time [data]")
			raw_input("Press any key to continue...")

	# Add time-jumps now
	if (verb >= VERB_DEBG):
		print "Found %d time jumps" % sum(np.abs(tjump) > 0.05)
	jumpidx = fjumpidx + tjumpidx
	
	# Inspect time jumps manually
	if (False and plot & PLOT_INTERACTIVE):
		for jump in np.argwhere(jumpidx):
			print "Time jump @ %d (showing %d--%d)" % (jump, max(jump-25, 0), jump+25)
			plt.figure(40)
			plt.clf()
			fdat_left = mflux[max(jump-25, 0):jump]
			fdat_right = mflux[jump:jump+25]
			ftime_left = mtime[max(jump-25, 0):jump]
			#np.arange(len(fdat_left)) - len(fdat_left)
			ftime_right = mtime[jump:jump+25]
			#np.arange(len(fdat_right))
			jump_size = (np.mean(fdat_left) - np.mean(fdat_right))
			jump_size /= np.min([fdat_left.std(), fdat_right.std()])
			plt.title(GLOB_ARGS.tgtname + " time jump @ %d = %g" % (jump, jump_size))
			plt.plot(ftime_left, fdat_left)
			plt.plot(ftime_right, fdat_right)
			plt.ylabel("Flux")
			plt.xlabel("Time [days]")
			raw_input("Press any key to continue...")

	# Group data based on these jumps. cumsum() will add 1 each time jumpidx 
	# is True, giving a 0-based index for each group
	groupids = jumpidx.cumsum()
	
	# Output flux goes here	
	fluxoff = np.zeros(mflux.shape)
	
	# Plot spline fits
	if (plot & PLOT_INTERACTIVE):
		plt.figure(21)
		plt.clf()
		plt.title(GLOB_ARGS.tgtname + " %d chunks poly/spline fit" % len(np.unique(groupids)))
		plt.ylabel("Flux")
		plt.xlabel("Time [days]")

	# Loop over groups, fit data
	for gid in np.unique(groupids):
		idx = np.argwhere(groupids==gid).ravel()
		n = len(idx)
		midx = np.intersect1d(idx, maskidx)
		if (verb >= VERB_DEBG): print "Working on group=%d, n=%d" % (gid, n)

		# Calculate median
		if (n < 10): 
			fluxoff[idx] = np.median(mflux[idx])
		# Calculate polynomial
		elif (n < 200):
			fit_parms = np.polyfit(mtime[midx], mflux[midx], deg=opoly)
			smooth_fit = np.poly1d(fit_parms)

			if (plot & PLOT_INTERACTIVE):
				plt.plot(mtime[idx], mflux[idx], ',')
				plt.plot(mtime[midx], mflux[midx], '.')
				plt.plot(mtime[idx], smooth_fit(mtime[idx]), '-')
				plt.plot(mtime[idx][::10], smooth_fit(mtime[idx][::10]), 'o')

			# Store this offset
			fluxoff[idx] = smooth_fit(mtime[idx])

		# Calculate spline
		else:
			# Spline fit with one knot per transit, (approx 30 datapts)
			d_phase = phase[idx][1:] - phase[idx][:-1]
			transit0 = mtime[idx][np.argwhere(d_phase < 0)+1].ravel()
			cadence = np.median(transit0[1:] - transit0[:-1])
			knots = transit0#np.arange(len(transit0))*cadence + transit0[0]
			
			# if knots are too close to the boundary, remove them
			if (knots[0] <= mtime[midx][10]): knots = knots[1:]
			if (knots[-1] >= mtime[midx][-10]): knots = knots[:-1]
			
			smooth_spl = LSQUnivariateSpline(mtime[midx], mflux[midx], t=knots, w=1.0/fluxerr[midx], k=ospline)
			
			# Store this offset
			fluxoff[idx] = smooth_spl(mtime[idx])

			if (plot & PLOT_INTERACTIVE):
				plt.plot(mtime[idx], mflux[idx], ',')
				plt.plot(mtime[midx], mflux[midx], '.')
				plt.plot(mtime[idx], smooth_spl(mtime[idx]), '-')
				plt.plot(knots, smooth_spl(knots), '*')
				#plt.plot(mtime[idx], smooth_spl(mtime[idx]), ':')

	if (plot & PLOT_INTERACTIVE):
		raw_input("Press any key to continue...")
	
	# Return data fit
	return fluxoff

#=============================================================================
# LIGHT CURVE MODEL
#=============================================================================

def transit_model_dp7(phase, sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, opoly=2, method=0, verb=0, plot=0):
	"""
	Disintegrating planet light curve modeling assuming an optically thin 
	cloud. All calculations are done in one dimension (azimuth).

	Last change: 29 Jun 2012. 
	
	Transcoded from dp7.pro (Matteo Brogi) to Python (Tim van Werkhoven)

	Old default params: sr=20.865*(1.-0.63**2.)**0.5, ep=5.1, ca=0.03, g=0.874, om=0.65, nmodel=400, opoly=2
	New default params: sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654
	
	@param [in] phase Vector of phases
	@param [in] sr Crossed stellar path in degrees
	@param [in] ep Exponential parameter for extinction drop-off
	@param [in] ca Extinction cross-section in units of stellar area
	@param [in] g Asymmetry parameter (-1.0 .. +1.0)
	@param [in] om Single-scattering albedo (0 .. 1)
	@param [in] nmodel Array size for calculating model
	@param [in] opoly Polynomial interpolation order
	@return Light curve with len(phase) points
	"""
	
	n = len(phase)
	
	# NOTES - Precalculated quatities
	# r = 0.65*R_sun
	# a = 0.013AU in meters
	# ra = arcsin(r/a)
	# sra = sr in radians

	#nmodel = 400                              # Light curve sampling
	binsz = 2.*np.pi/float(nmodel)             # Bin size
	azim = np.arange(nmodel)*binsz             # Azimuth vector
	azim_ph = azim/(2.0*np.pi) - 0.5           # Azimuth as phase
	r = 4.52e8                                 # Stellar radius in meters
	a = 1.95e9                                 # Semimajor axis in meters
	sra = sr*0.0174533                         # Half crossed chord in radians

	# Cloud 
	cloud = ca*np.exp(-ep*azim)

	# Stop midway to cross-check with Matteo's model
	if (method == 1): return cloud

	# Star - Only the non-zero part of the stellar profile is computed,
	#        in order to speed up the convolution.
	
	nstar = int(math.floor(sra/binsz) + 1)
	subaz = azim[np.arange(nstar)]
	star = 1.-0.79*(1.-a*np.sqrt(math.sin(sra)**2.-np.sin(subaz)**2.)/r)
	star[nstar-1] = 0.21*(sra/binsz-math.floor(sra/binsz))
	star = np.hstack([star[:0:-1],star])

	# Normalization - For zero impact parameter, the integral of the
	#                 stellar profile has to be 1. For non-zero impact
	#                 parameter, the integral is scaled by the factor
	#                 (chord/2.*radius)^2, determined by fitting 
	#                 pre-calculated limb-darkened stellar profiles.
	
	star = star/np.sum(star)*(a*math.sin(sra)/r)**2.

	# Stop midway to cross-check with Matteo's model
	if (method == 2): return star

	# Light curve of star - dust extinction

	#lc0 = 1. - np.convolve(np.roll(cloud, len(cloud)/2), star, mode='same',)
	lc0 = 1. - np.convolve(np.roll(cloud, len(cloud)/2), star, mode='same',)
	lc0 = np.roll(lc0, len(lc0)/2)

	# Stop midway to cross-check with Matteo's model
	if (method == 3): return lc0
	
	if (plot):
		plt.figure(40); plt.clf()
		plt.plot(azim_ph, cloud, '--', label='cloud')
		plt.plot(azim_ph[np.arange(len(star))], star, '--', label='star')
		plt.legend()
		plt.title("Star & dust")
		
		plt.figure(42); plt.clf()
		plt.plot(azim_ph, lc0)
		plt.title("Light curve")
		if (plot & PLOT_INTERACTIVE):
			raw_input("Press any key to continue...")
	
	# Scattering part: HG function normalized by the solid angle.
	
	scatt = (1.-g**2.)/((1.-2.*g*np.cos(azim)+g**2.)**1.5)/(4.*np.pi)
	scatt[nmodel/2-nstar+1:nmodel/2+nstar] = 0.       #Secondary eclipse

	# Stop midway to cross-check with Matteo's model
	if (method == 4): return scatt
	
	# Final light curve = star - extinction + scattering.
	# The multiplicative factor is !pi*(r_star/a)^2. = 0.169, where d
	# r_star = 0.65 * 6.955*10^8m
	# a = 0.013*149597870700
	# !pi*(r_star/a)^2. = 0.16976
	# is the semimajor axis of the orbit and r_star the stellar radius.

	lc_scatt = 0.16976 * om * sp.ndimage.filters.convolve(cloud, scatt, mode='wrap')*binsz
	lc_scatt = np.roll(lc_scatt, len(lc_scatt)/2)
	lc1 = lc0 + lc_scatt
	
	# Stop midway to cross-check with Matteo's model
	if (method == 51): return lc_scatt
	if (method == 5): return lc1

	if (plot):
		plt.figure(43); plt.clf()
		plt.title("Scattering component")
		plt.plot(azim_ph, scatt, label='scatt')
		plt.plot(azim_ph, lc_scatt, label='lc_scatt')
		plt.plot(azim_ph, cloud, label='cloud')
		plt.legend(loc='upper right')
		
		plt.figure(42); plt.clf()
		plt.plot(azim_ph, lc0, label='lc')
		plt.plot(azim_ph, lc1, label='lc+dust')
		plt.legend(loc='lower right')
		plt.title("Light curve")
		if (plot & PLOT_INTERACTIVE):
			raw_input("Press any key to continue...")
	
	# Convolve with exposure time (dt = 29.4 minutes). Half of the box
	# width is given by dt/period*!pi = 0.098225 radians.
	# period = 15.6854 hr
	# dt = 29.4/60 hr
	# dt/period = (29.4/60)/15.6854*np.pi = 0.09814097
	
	nwin = math.floor(0.09814097/binsz)
	win = np.ones(nwin+1, dtype=np.float)
	win[nwin] = 0.09814097/binsz - nwin
	win = np.hstack([win[:0:-1], win])
	win = win/sum(win)

	# Convolve data with exposure time
	#lc2 = np.convolve(lc1, win, mode='same')
	lc2 = sp.ndimage.filters.convolve(lc1, win, mode='wrap')
	# lc2[:nwin+1] = lc1[:nwin+1]
	# lc2[-nwin:] = lc1[-nwin:]
	lc2 = np.roll(lc2, len(lc2)/2)

	# Stop midway to cross-check with Matteo's model
	if (method == 6): return lc2

	if (plot):
		plt.figure(44); plt.clf()
		plt.title("Exposure convolution")
		plt.plot(win, label='exposure')
		plt.legend(loc='lower right')

		plt.figure(42); plt.clf()
		plt.plot(azim_ph, lc0, label='lc')
		plt.plot(azim_ph, lc1, label='lc+dust')
		plt.plot(azim_ph, lc2, ':o', label='lc+dust+conv')
		plt.title("Light curve")
		plt.legend(loc='lower right')
		if (plot & PLOT_INTERACTIVE):
			raw_input("Press any key to continue...")

	# If phase is not given, return the raw lightcurve:	
	if (not len(phase)):
		return lc2

	# Interpolation to the input phase vector
	lc_intp = sp.interpolate.interp1d(azim_ph, lc2, kind=[None, 'linear', 'quadratic', 'cubic'][opoly])

	#lc_intp = np.poly1d(np.polyfit(azim, lc2, deg=opoly))
	return lc_intp(np.asanyarray(phase))

