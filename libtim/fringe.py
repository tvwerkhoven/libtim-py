#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file fringe.py -- fringe analysis library
@author Tim van Werkhoven, from pyfa.py, based on Christoph Keller's fa.gap from 2001
@date 20120504
@copyright Copyright (c) 2011--2013 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)

Library for fringe patterns processing and wavefront extraction.

# References

[1] "Analysis of the phase unwrapping algorithm Applied Optics", Vol. 21 (July 1982), 2470, doi:10.1364/ao.21.002470 by K. Itoh
[2] "Interferometry mathematics, algorithms and data" (10 February 2010), pp. 1-50 by Michael Peck http://home.earthlink.net/~mlpeck54/astro/imath/imath.pdf
[3] http://www.mathworks.com/products/signal/demos.html?file=/products/demos/shipping/signal/hilberttransformdemo.html&nocookie=true
[4] http://w3.msi.vxu.se/exarb/mj_ex.pdf
[5] https://ccrma.stanford.edu/~jos/st/Analytic_Signals_Hilbert_Transform.html
[6] "Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry" Journal of the Optical Society of America, Vol. 72, No. 1. (1 January 1982), pp. 156-160, doi:10.1364/josa.72.000156 by Mitsuo Takeda, Hideki Ina, Seiji Kobayashi

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

# Import libs
import numpy as np
import scipy.signal
import sys, os
import time
from os.path import join as pjoin

# Import my own utilities
import libtim as tim
import libtim.im
import libtim.zern
import libtim.util
import libtim.file
import libtim.fft
import libtim.xcorr
import libtim.shwfs

from unwrap import flood_quality

def sim_fringe(phase, cfreq, noiseamp=0, phaseoffset=0, noisesmooth=10):
	"""
	Simulate fringe pattern for **phase** interfering with flat 
	reference beam with carrier frequency **cfreq**.

	Based on "Fourier-transform method of fringe-pattern analysis for 
	computer-based topography and interferometry" Journal of the Optical 
	Society of America, Vol. 72, No. 1. (1 January 1982), pp. 156-160, 
	doi:10.1364/josa.72.000156 by Mitsuo Takeda, Hideki Ina, Seiji Kobayashi

	@param [in] phase 2D array of phase [radians]
	@param [in] cfreq 2D carrier frequency of reference wave [cycles/axis]
	@param [in] noiseamp Amplitude of normal-random noise to add
	@param [in] phaseoffset Phase offset for fringe patterns.
	@return 2D real fringe pattern
	"""

	# Make position grid, position[I,J] = (x,y)
	position = np.indices(phase.shape)*1./np.r_[phase.shape].reshape(-1,1,1)

	cfreq = np.array(cfreq).reshape(-1,1,1)
	fringe = np.sin(2*np.pi*(cfreq*position).sum(0) + phaseoffset + phase)
	fnoise = np.random.randn(*fringe.shape)

	if (noisesmooth):
		smooth = tim.im.mk_rad_mask(noisesmooth)<1
		fnoise = scipy.signal.fftconvolve(fnoise, smooth.astype(int), mode='same')

	fnoise *= noiseamp/fnoise.std()

	return fringe+fnoise

def fringe_cal(refimgs, wsize=-0.5, cpeak=0, do_embed=True, method='parabola', store_pow=True, ret_pow=False, outdir='./'):
	"""
	Calibrate fringe analysis here using a reference frame with pure carrier
	fringes. This frame will be used to calculate the carrier frequency such
	that subsequent images can be analysed around this frequency.

	@bug Sometimes carrier frequencies found are *inside* the central blanked out area?

	@param [in] refimgs List of reference fringe patterns
	@param [in] wsize Apodisation window size
	@param [in] cpeak Size of central peak in FT power that should be ignored
	@param [in] do_embed Embed data before FFT. This improves sideband localisation.
	@param [in] method Method to determine the sideband center, see locate_sb
	@param [in] store_pow Store the Fourier power spectrum
	@param [in] ret_pow Return Fourier power spectra
	@param [in] outdir Directory to store output to
	@return len(reflist) x 2 ndarray of (freq0, freq1) spatial frequency pairs. If **ref_pow**, return list of power spectra as well
	"""

	cfreql = []
	fftpowl = []
	fftpowzooml = []

	fft_mask = tim.fft.mk_apod_mask(refimgs[0].shape, wsize=wsize, shape='rect', apod_f='cosine')

	for idx, refimg in enumerate(refimgs):
		postf = "_%d" % (idx)

		refimg_apod = (refimg - refimg.mean()) * fft_mask
		if (do_embed):
			refimg_apod = tim.fft.embed_data(refimg_apod, direction=1)

		# Fourier transform <refimg>, calculate power spectrum. This gives 
		# streaking as seen in log scaling, but this not interfere with
		# sideband localisation. Padding the data with zeroes does *not* 
		# seem to solve this.
		refimg_ft = tim.fft.descramble(np.fft.fft2(refimg_apod))
		if (do_embed):
			refimg_ft = tim.fft.embed_data(refimg_ft, direction=-1)

		refimg_pow = np.abs(refimg_ft**2.0)
		carr_freq = locate_sb(refimg_pow, cpeak=cpeak, method=method)

		# Zoom region: 1/6 of the radius (i.e. at least 6 pixels per fringe)
		sh = refimg_pow.shape
		npix = 12. / (1+int(do_embed))
		fft_pow_zoom = refimg_pow[int(sh[0]/2.-sh[0]/npix):int(sh[0]/2.+sh[0]/npix), int(sh[1]/2.-sh[1]/npix):int(sh[1]/2.+sh[1]/npix)]

		if (store_pow):
			tim.file.store_file(pjoin(outdir, 'fa_cal_pow_zoom'+postf+'.png'), np.log10(fft_pow_zoom))

		fftpowl.append(refimg_pow)
		fftpowzooml.append(fft_pow_zoom)

		# If we embedded the data, the frequency we find is twice as high 
		# because the input array was larger, correct this here
		if (do_embed):
			carr_freq /= 2.0

		cfreql.append(carr_freq)

	if (ret_pow):
		return np.r_[cfreql], fftpowl, fftpowzooml
	else:
		return np.r_[cfreql]

def locate_sb(fftpow, cpeak=None, method='parabola'):
	"""
	Given a (descrambled) Fourier power spectrum **fftpow**, find the 
	location of the (first) side-band peak.

	**cpeak** defines the radius of the central region to ignore to prevent 
	conflicts with possible central maxima. If not set, this will be 
	determined automatically by looking at the radial intensity profile and 
	locating the first minimum. The radius at which this minimum lies will 
	be used as blocking filter.

	@param [in] fftpow Descrambled power spectrum to analyze
	@param [in] cpeak Radius of central region to ignore. If None, find ourselves
	@param [in] method Method to determine the sideband center, either 'parabola' for a 2D parabolic fit or 'cog' for center of gravity. The former is slightly more precise, the latter is more robust
	@return Numpy array of (x, y) subpixel maximum of first sideband
	"""

	if (cpeak < 0):
		raise ValueError("negative <cpeak> is illegal!")

	# Detect central peak if not given
	if (not cpeak):
		rad_prof = tim.im.mk_rad_prof(fftpow)
		# Detect where intensity stops declining. Skip first pixel because this is already low due to removing the mean from the FFT
		d_rad_prof = rad_prof[1:-1] - rad_prof[2:]
		cpeak = np.argwhere(d_rad_prof < 0)[0][0]

	# Ignore center <cpeak> pixels in further processing
	sz = np.r_[fftpow.shape]
	rad_mask = tim.im.mk_rad_mask(*sz, norm=False)
	fftpow[rad_mask <= cpeak] = 0

	# Find sub-pixel maximum around brightest pixel
	if (method == 'parabola'):
		# @todo Perhaps this should be a wider center of gravity search, 9 pixels is a bit too small for real data
		sb_loc = tim.xcorr.calc_subpixmax(fftpow, sz/2., index=True)
	elif (method == "cog"):
		# Select half image so we exclude the other sideband, otherwise the
		# CoG will go to the origin.
		max0, max1 = np.argwhere(fftpow == fftpow.max())[0]
		dmax0, dmax1 = (max0, max1) - sz/2

		# Crop a region around maximum, never include the origin, take a 
		# symmetric crop region in both axes. Make sure the maximum is in 
		# the center
		r0 = np.min([abs(dmax0), abs(dmax0), sz[0]/20, sz[1]/20])
		fftpow_crop = fftpow[max0-r0:max0+r0+1, max1-r0:max1+r0+1]
		# From this crop region, calculate the CoG
		crop_mid = tim.shwfs.calc_cog(fftpow_crop, index=True)
		# Give maximum back in coordinates of fftpow 
		sb_loc = np.r_[max0-r0, max1-r0] - sz/2. + crop_mid
	else:
		raise ValueError("Unknown method (use 'parabola' or 'cog')")

	# There are always two symmetrical sidebands, always find the one with the
	# highest sum to ensure we always find the same.
	# (-15, 25) and (15, 25) will work
	# (-5, 5) and (5, -5) will not work
	# (0, 10) and (0, -10) will work
	# (-0.001, 10), (0, 10) and (0, -10) will work
	## @bug Fix me, this will not work always
	if (np.sum(-sb_loc) > np.sum(sb_loc)):
		return -sb_loc

	return sb_loc

def filter_sideband(img, cfreq, sbsize, method='spectral', apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', do_embed=True, cache={}, ret_pow=False, get_complex=False, verb=0):
	"""
	Filter out sideband from a real image, return phase and amplitude. Phase
	is returned in radians, complex components are given 'as-is'.

	For extracting the sideband we have three options:
	1. Shift carrier frequency to origin using spectral shifting
	2. Select a passband around the carrier frequency in Fourier space
	3. Select a circular frequency region around the carrier frequency in Fourier space (work in progress)

	# Spectral method (recommended):
	1. Shift frequencies in image space such that carrier frequency is at the origin
	2. Apodise image and Fourier transform
	3. Lowpass filter around frequency zero. Due to spectral shifting, this selects our carrier frequency.
	4. Inverse transform, get complex components
	5. Calculate phase and unwrap

	# Passband method (not recommended):
	1. Apodise image and Fourier transform
	2. Bandpass filter around carrier frequency
	3. Half-plane cut and inverse transform
	4. Calculate phase and unwrap
	5. Remove carrier frequency tilt in phase

	# Circular method: (WIP)
	1. Apodise image and Fourier transform
	2. Apply circular filter around carrier frequency
	3. Inverse transform single sideband
	4. Calculate phase and unwrap
	5. Remove carrier frequency tilt in phase

	@param [in] img CCD image to process
	@param [in] cfreq Carrier frequency in cycles/frame. If 'None', remove all tip-tilt from output phase.
	@param [in] sbsize The size of the sideband extraction region, as fraction of the magnitude of cfreq
	@param [in] method Extraction method to use, one of ('spectral', 'bandpass')
	@param [in] apt_mask Boolean mask to select ROI
	@param [in] unwrap Unwrap phase
	@param [in] wsize Window size for apodisation mask, see tim.fft.mk_apod_mask
	@param [in] wfunc Window function for apodisation mask, see tim.fft.mk_apod_mask
	@param [in] cache Will be filled with cached items. Re-supply next call to speed up process
	@param [in] ret_pow Return Fourier power around **cfreq** as well.
	@param [in] verb Verbosity
	@returns Tuple of (phase [rad], amplitude) as numpy.ndarrays
	"""

	cfreq = np.asanyarray(cfreq)

	if (method == 'spectral'):
		# 1a. Calculate shift matrix
		if cache.has_key('spec_sshift'):
			sshift = cache['spec_sshift']
		else:
			slope = np.indices(img.shape, dtype=np.float) / np.r_[img.shape].reshape(-1,1,1)
			slope = np.sum(slope * cfreq.reshape(-1,1,1), 0)
			sshift = np.exp(-1j * 2 * np.pi * slope)
			cache['spec_sshift'] = sshift

		# 1b. Shift image after removing the mean
		img_sh = (img - img.mean()) * sshift

		# 2. Apodise image and FFT
		if cache.has_key('spec_apodmask'):
			apod_mask = cache['spec_apodmask']
		else:
			apod_mask = tim.fft.mk_apod_mask(img.shape, wsize=wsize, shape='circ', apod_f=wfunc)
			cache['spec_apodmask'] = apod_mask

		img_apod = img_sh * apod_mask
		if (do_embed):
			img_apod = tim.fft.embed_data(img_apod, direction=1)

		img_sh_ft = tim.fft.descramble(np.fft.fft2(img_apod))

		# 3. Lowpass filter and crop
		lowpass = (1+do_embed) * sbsize * (np.r_[cfreq]**2.0).sum()**0.5
		if cache.has_key('spec_lowpassmask'):
			lowpass_mask = cache['spec_lowpassmask']
		else:
			lowpass_mask = tim.fft.mk_apod_mask(img_sh_ft.shape, apodsz=lowpass*2, shape='circle', wsize=-.5, apod_f=wfunc)
			cache['spec_lowpassmask'] = lowpass_mask
		img_sh_filt = img_sh_ft * lowpass_mask

		if (ret_pow):
			sz = img_sh_filt.shape
			fftpow = np.abs(img_sh_filt[sz[0]/2-lowpass:sz[0]/2+lowpass,
				sz[1]/2-lowpass:sz[1]/2+lowpass].copy())**2.0

		# 4. IFFT, get complex components
		img_ifft = np.fft.ifft2(tim.fft.descramble(img_sh_filt, -1))
		if (do_embed):
			img_ifft = tim.fft.embed_data(img_ifft, direction=-1)

		# 4b. Sometimes we only need the complex components
		if (get_complex):
			return img_ifft

		# 5. Calculate phase and unwrap
		phase_wr = np.arctan2(img_ifft.imag, img_ifft.real)
		amp = np.abs(img_ifft**2.0)

		if (np.any(apt_mask)):
			phase_wr *= apt_mask
			amp *= apt_mask

		if (unwrap): phase = flood_quality(phase_wr, amp)
		else: phase = phase_wr

		if (np.any(apt_mask)):
			phase -= phase[apt_mask].mean()
			phase *= apt_mask
		else:
			phase -= phase.mean()

	elif (method == 'passband'):
		# 1. Apodise data, Fourier transform
		if cache.has_key('pass_apodmask'):
			apod_mask = cache['pass_apodmask']
		else:
			apod_mask = tim.fft.mk_apod_mask(img.shape, wsize=wsize, apod_f=wfunc)
			cache['pass_apodmask'] = apod_mask

		img_apod = (img - img.mean()) * apod_mask
		img_ft = tim.fft.descramble(np.fft.fft2(img_apod))

		# 2. Bandpass filter around carrier frequency
		if cache.has_key('pass_filtmask'):
			filt_mask = cache['pass_filtmask']
		else:
	 		fa_inner = sbsize * (np.r_[cfreq]**2.0).sum()**0.5
	 		fa_outer = 1.0/sbsize * (np.r_[cfreq]**2.0).sum()**0.5
			filt_mask_in = 1.0-tim.fft.mk_apod_mask(img_ft.shape, apodsz=fa_inner*2, shape='circle', wsize=-.5, apod_f=wfunc)
			filt_mask_out = tim.fft.mk_apod_mask(img_ft.shape, apodsz=fa_outer*2, shape='circle', wsize=-.5, apod_f=wfunc)
	 		filt_mask = filt_mask_in * filt_mask_out
	 		cache['pass_filtmask'] = filt_mask

		img_filt = img_ft * filt_mask

		if (ret_pow):
			sz = img_filt.shape
			fa_outer = 1.0/sbsize * (np.r_[cfreq]**2.0).sum()**0.5
			fftpow = np.abs(img_filt[sz[0]/2:sz[0]/2+fa_outer,
				sz[1]/2:sz[1]/2+fa_outer].copy())**2.0

		# 3. Half-plane cut and inverse transform
		## @todo How to handle odd image sizes?
		imgsh = img_filt.shape
		if (imgsh[0] % 2 == 0):
			if (cfreq[0] > 0):
				img_filt[:imgsh[0]/2] = 0
			else:
				img_filt[imgsh[0]/2:] = 0
		elif (imgsh[1] % 2 == 0):
			if (cfreq[1] > 0):
				img_filt[:, :imgsh[1]/2] = 0
			else:
				img_filt[:, imgsh[1]/2:] = 0
		else:
			img_filt[:, :imgsh[1]/2] = 0
		
		# 4. IFFT, get complex components
		img_ifft = np.fft.ifft2(tim.fft.descramble(img_filt, -1))

		# 4b. Sometimes we only need the complex components
		if (get_complex):
			return img_ifft

		# 5. Calculate phase and unwrap
		phase_wr = np.arctan2(img_ifft.imag, img_ifft.real)
		amp = np.abs(img_ifft**2.0)

		if (np.any(apt_mask)):
			phase_wr *= apt_mask
			amp *= apt_mask

		if (unwrap): phase = flood_quality(phase_wr, amp)
		else: phase = phase_wr

		# 5. Calculate slope to subtract
		## @todo This can be improved by using orthogonal vectors
		if cache.has_key('pass_slope'):
			slope = cache['pass_slope']
		else:
			slope = np.indices(phase.shape, dtype=np.float) / np.r_[phase.shape].reshape(-1,1,1)
			slope = np.sum(slope * np.round(cfreq).reshape(-1,1,1), 0)
			slope -= slope.mean()
			cache['pass_slope'] = slope
		
		phase -= slope

		if (np.any(apt_mask)):
			phase -= phase[apt_mask].mean()
			phase *= apt_mask
		else:
			phase -= phase.mean()
	elif (method == 'circular'):
		# circular method:
		# - Apodise, Fourier transform
		# - Apply circular filter around carrier frequency
		# - Inverse transform single sideband
		# - Remove carrier frequency tilt in phase
		raise RuntimeWarning("Work in progress")

		apod_mask = tim.fft.mk_apod_mask(img.shape, wsize=wsize, apod_f=wfunc)
		img_apod = (img - img.mean()) * apod_mask

		tim.im.inter_imshow(img_apod, doshow=verb%100>=30, desc="circular::apod img")

		img_ft = tim.fft.descramble(np.fft.fft2(img_apod))

		# Circular-band filter, radius of half the carrier frequency
 		band_rad = 0.5 * (np.r_[cfreq]**2.0).sum()**0.5
 		band_pos = np.r_[cfreq]

		filt_mask = tim.fft.mk_apod_mask(img_ft.shape, apodpos=band_pos, apodsz=band_rad*2, shape='circle', wsize=-.5, apod_f=wfunc)

		img_filt = img_ft * filt_mask
		if (ret_pow):
			fftpow = None
		phase = None
		amp = None
	else:
		raise ValueError("Unknown method '%s'" % (method))

	if (ret_pow):
		return (phase, amp, fftpow)
	else:
		return (phase, amp)

def get_dark_flat(flats, darks, roi=(0,-1,0,-1)):
	"""
	Read dark- and flatfields in the lists **flats** and **darks**, convert to float, and return the average for both lists as a tuple.

	@param [in] flats List of paths for flatfields
	@param [in] darks List of paths for darkfields
	@param [in] roi Region of interest
	@return Tuple of (flatfield average, darkfield average) as float arrays
	"""

	flimg = None
	dkimg = None
	if (flats or darks):
		if (flats):
			flimg = sum( [tim.file.read_file(f, roi=roi).astype(float) for f in flats] ) * (1.0/len(flats))
		if (darks):
			dkimg = sum( (tim.file.read_file(f, roi=roi).astype(float) for f in darks) ) * (1.0/len(darks))

	return flimg, dkimg

def avg_phase(wavecomps, ampweight=False):
	"""
	Given a list of complex wave components **wavecomps**, average these 
	phasor-wise. Returned phase is in radians.

	We first rotate the complex phasors to have the same angle for the 
	center element, then we average the the phases and take the arctangent, 
	resulting in the phase.

	@param [in] wavecomps Iterable of arrays of complex wave components.
	@param [in] ampweight Weight phase averaging by amplitude
	@return Tuple of (weighted) mean phase (in rad) and mean amplitude
	"""

	mid = np.r_[wavecomps[0].shape]/2

	# Rotate phasors such that the center element has zero rotation
	wc_rot = np.zeros_like(wavecomps)
	for wc, wcr in zip(wavecomps, wc_rot):
		angle = -np.angle(wc[mid[0], mid[1]])
		wcr.real = wc.real*np.cos(angle) - wc.imag*np.sin(angle)
		wcr.imag = wc.real*np.sin(angle) + wc.imag*np.cos(angle)

	# Compute phases from rotated complex wave components.
	if (ampweight):
		wc_rot_avg = np.average(wc_rot, axis=0, weights=np.abs(wc_rot**2.0).mean(1).mean(1))
	else:
		wc_rot_avg = np.average(wc_rot, axis=0)

	return np.arctan2(wc_rot_avg.imag, wc_rot_avg.real), np.abs(wc_rot_avg**2.0)

def phase_grad(wave, wrap=0, clip=0, apt_mask=slice(None), asvec=False):
	"""
	Calculate gradient of phase (in rad), pixel by pixel. The gradient is 
	defined as:

		g_i = (ph_i - ph_i-1)

	where **g_i** is the gradient, and **ph_i** is the phase in pixel i. The 
	first row of pixels is always 0 because the gradient is undefined there.

	**Warning**: do not input data which has been cropped by an aperture 
	mask already, as this will give very sharp gradient edges. Instead, give 
	the full phase and mask the gradient with **apt_mask**.

	@param [in] wave Input phase [rad]
	@param [in] wrap Wrap per-pixel phase variation by this value
	@param [in] clip Clip gradient to this value
	@param [in] apt_mask Return only data inside this boolean mask (only for asvec!)
	@param [in] asvec Return as vector
	@return Tuple of (grad0, grad0)
	"""

	# Add line of zeros for undefined gradients
	dwave = wave[1:, :] - wave[:-1, :]
	dwave0 = np.vstack([np.zeros_like(wave[0:1]), dwave])
	dwave = wave[:, 1:] - wave[:, :-1]
	dwave1 = np.hstack([np.zeros_like(wave[:,0:1]), dwave])

	if (wrap):
		# @todo: >wrap should be >pi, wrap offset should be 2 pi (?)
		dwave0[dwave0>wrap] = dwave0[dwave0>wrap] - wrap
		dwave0[dwave0<wrap] = dwave0[dwave0<wrap] + wrap
		dwave1[dwave1>wrap] = dwave1[dwave1>wrap] - wrap
		dwave1[dwave1<wrap] = dwave1[dwave1<wrap] + wrap

	if (clip):
		dwave0 = np.clip(dwave0, -clip, +clip)
		dwave1 = np.clip(dwave1, -clip, +clip)

	if (asvec):
		return np.hstack([dwave0[apt_mask].ravel(), dwave1[apt_mask].ravel()])
	else:
		return dwave0, dwave1

### EOF

