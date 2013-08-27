#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file test_fringe.py
@brief Test libtim/fringe.py library
@author Tim van Werkhoven
@date 20130619
@copyright Copyright (c) 2013 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
Testcases for fringe.py library.

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

# Import local libs
from fringe import *
import shwfs

# Import other libs
from timeit import Timer
import unittest
import numpy as np
import pylab as plt
import logging
logging.basicConfig( stream=sys.stderr )
logging.getLogger( "test_fringe" ).setLevel( logging.DEBUG )

SHOWPLOTS=True
TESTDATAPATH=pjoin(os.path.dirname(__file__), "test_fringe_in/")

class TestFringecal(unittest.TestCase):
	# fringe_cal(refimgs, wsize=-0.5, cpeak=0, do_embed=True, store_pow=True, ret_pow=False, outdir='./'):
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )
		self.cflst = [(3, 4), (18.3, 1.3), (22.22, 11.11)]
		self.sz = (320, 240)

	def test0a_cal(self):
		"""Test function calls"""
		for cfreq in self.cflst:
			fpatt = sim_fringe(np.zeros(self.sz), cfreq)
			
			cfreq1 = fringe_cal([fpatt], store_pow=False, do_embed=False)[0]
			self.assertAlmostEqual(sum(cfreq1), sum(cfreq), places=0)
			np.testing.assert_almost_equal(cfreq1, cfreq, decimal=0)

			cfreq1 = fringe_cal([fpatt, fpatt], store_pow=False, do_embed=False)[1]
			self.assertAlmostEqual(sum(cfreq1), sum(cfreq), places=0)
			np.testing.assert_almost_equal(cfreq1, cfreq, decimal=0)

			cfreq1, fftpl, fftpzooml = fringe_cal([fpatt], store_pow=False, ret_pow=True, do_embed=False)
			self.assertAlmostEqual(sum(cfreq1.mean(0)), sum(cfreq), places=0)
			np.testing.assert_almost_equal(cfreq1.mean(0), cfreq, decimal=0)

			cfreq1, fftpl, fftpzooml = fringe_cal([fpatt, fpatt, fpatt], store_pow=False, ret_pow=True, do_embed=False)
			self.assertAlmostEqual(sum(cfreq1.mean(0)), sum(cfreq), places=0)
			np.testing.assert_almost_equal(cfreq1.mean(0), cfreq, decimal=0)

	def test1a_cal_flat(self):
		"""Generate fringe pattern for flat phase"""
		for cfreq in self.cflst:
			fpatt = sim_fringe(np.zeros(self.sz), cfreq)
			cfreq1a = fringe_cal([fpatt], store_pow=False, do_embed=False)[0]
			cfreq1b = fringe_cal([fpatt], store_pow=False, do_embed=False, method='cog')[0]
			cfrata = 1.-np.r_[cfreq]/cfreq1a
			cfratb = 1.-np.r_[cfreq]/cfreq1b
			self.log.debug("ref: %.3f, %.3f, rec1: %.3f, %.3f, rec2: %.2g, %.2g", cfreq[0], cfreq[1], cfreq1a[0], cfreq1a[1], cfreq1b[0], cfreq1b[1])
			self.assertAlmostEqual(np.sum(cfrata), 0, places=0)
			self.assertAlmostEqual(np.sum(cfratb), 0, places=0)
			np.testing.assert_almost_equal(cfreq1a, cfreq, decimal=0)
			np.testing.assert_almost_equal(cfreq1a, cfreq, decimal=0)

	def test1b_cal_flat_embed(self):
		"""Generate fringe pattern for flat phase, embed"""
		for cfreq in self.cflst:
			fpatt = sim_fringe(np.zeros(self.sz), cfreq)
			cfreq1 = fringe_cal([fpatt], store_pow=False, do_embed=True)[0]
			cfrat = 1.-np.r_[cfreq]/cfreq1
			self.log.debug("ref: %.3f, %.3f, rec: %.3f, %.3f, ratio: %.2g, %.2g", cfreq[0], cfreq[1], cfreq1[0], cfreq1[1], cfrat[0], cfrat[1])
			self.assertAlmostEqual(sum(cfreq1), sum(cfreq), places=1)
			np.testing.assert_almost_equal(cfreq1, cfreq, decimal=1)

	def test2a_cal_flat_noise(self):
		"""Generate fringe pattern for flat phase with noise"""
		for cfreq in self.cflst:
			fpatt = sim_fringe(np.zeros(self.sz), cfreq, noiseamp=0.5)
			cfreq1 = fringe_cal([fpatt], store_pow=False, do_embed=False)[0]
			cfrat = 1.-np.r_[cfreq]/cfreq1
			self.log.debug("ref: %.3f, %.3f, rec: %.3f, %.3f, ratio: %.2g, %.2g", cfreq[0], cfreq[1], cfreq1[0], cfreq1[1], cfrat[0], cfrat[1])
			self.assertAlmostEqual(sum(cfreq1), sum(cfreq), places=0)
			np.testing.assert_almost_equal(cfreq1, cfreq, decimal=0)

	def test2b_cal_flat_noise(self):
		"""Generate fringe pattern for flat phase with noise, embed"""
		for cfreq in self.cflst:
			fpatt = sim_fringe(np.zeros(self.sz), cfreq, noiseamp=0.5)
			cfreq1 = fringe_cal([fpatt], store_pow=False, do_embed=True)[0]
			cfrat = 1.-np.r_[cfreq]/cfreq1
			self.log.debug("ref: %.3f, %.3f, rec: %.3f, %.3f, ratio: %.2g, %.2g", cfreq[0], cfreq[1], cfreq1[0], cfreq1[1], cfrat[0], cfrat[1])
			self.assertAlmostEqual(sum(cfreq1), sum(cfreq), places=1)
			np.testing.assert_almost_equal(cfreq1, cfreq, decimal=1)

	def test3a_cal_real(self):
		"""Test quality of fringe_cal with real data"""
		files = ['fringe_130622_154235Z_000082_img.jpg',
			'fringe_130622_154235Z_000139_img.jpg',
			'fringe_130622_154235Z_000220_img.jpg',
			'fringe_130622_154235Z_000330_img.jpg',
			'fringe_130622_154235Z_000412_img.jpg',
			'fringe_130622_154235Z_000505_img.jpg']
		fringes = [tim.file.read_file(pjoin(TESTDATAPATH,f)) for f in files]
		sz = fringes[0].shape
		cfreq = (7.81111508,  24.76214802)

		do_embed = True
		cfreqs0, impowl, impowzml = fringe_cal(fringes, wsize=-0.5, cpeak=0, do_embed=do_embed, method='parabola', store_pow=False, ret_pow=True, outdir='./')
		cfreq0 = np.mean(cfreqs0, 0)
		cfreqs1 = fringe_cal(fringes, wsize=-0.5, cpeak=0, do_embed=do_embed, method='cog', store_pow=False, ret_pow=False, outdir='./')
		cfreq1 = np.mean(cfreqs1, 0)
		sz = np.r_[impowzml[0].shape]/(1+do_embed)
		cfreq2 = np.argwhere(impowzml[0] == impowzml[0].max())[0]/2 - sz/2
		
		for impowzm in impowzml:
			plt.figure(100); plt.clf()
			plt.title("Carr. freq. FFT power log zoom")
			plt.imshow(np.log10(impowzm), extent=(-sz[1]/2, sz[1]/2, -sz[0]/2, sz[0]/2))
			plt.plot(cfreq2[1], cfreq2[0], "+", label='max') # Maximum pixel
			plt.plot(cfreq0[1], cfreq0[0], "^", label='para') # Parabolic interpolation
			plt.plot(cfreq1[1], cfreq1[0], "*", label='cog') # Center of gravity
			plt.legend(loc='best')
			if (raw_input("Continue [b=break]...") == 'b'): break
		plt.close()

	def test3b_cal_qual(self):
		"""Test quality of fringe_cal with noise"""
		self.log.debug("cf: carr freq, noise: noise fractio; ratio1*e3: embedded cfreq quality, ratio2*e3: no embed cfreq quality")
		for idx, cfreq in enumerate(self.cflst):
			fpatt = sim_fringe(np.zeros(self.sz), cfreq)
			for noiseamp in np.linspace(0, 2, 3):
				fnoise = np.random.randn(*fpatt.shape)*noiseamp
				cfreq1 = fringe_cal([fpatt+fnoise], store_pow=False, do_embed=True)[0]
				cfreq2 = fringe_cal([fpatt+fnoise], store_pow=False, do_embed=False)[0]
				cfrat1 = (1.-np.r_[cfreq]/cfreq1)*1e3
				cfrat2 = (1.-np.r_[cfreq]/cfreq2)*1e3
				self.log.debug("cf: %d, noise: %#5.3g, r1*e3: %+#4.2f, %+#4.2f, r2*e3: %+#4.2f, %+#4.2f", idx, noiseamp, cfrat1[0], cfrat1[1], cfrat2[0], cfrat2[1])

class TestFiltersb(unittest.TestCase):
	# filter_sideband(img, cfreq, sbsize, method='spectral', apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', cache={}, ret_pow=False, verb=0)
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )
		self.cflst = [(3, 4), (18.3, 1.3), (22.22, 11.11)]
		self.sz = (320, 240)
		self.fcache = {}

	def test0_filtersb_test_nom(self):
		"""Test if filter_sideband() works nominally"""
		cf = self.cflst[-1]
		fpatt = sim_fringe(np.zeros(self.sz), cf)

		for method in ['spectral', 'passband']:
			phase, amp, ftpow = filter_sideband(fpatt, cf, 0.5, method=method, apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', ret_pow=True, verb=0)
			self.assertEqual(phase.shape, fpatt.shape)
			self.assertEqual(amp.shape, fpatt.shape)
			self.assertEqual(np.isnan(phase).sum(), 0)
			self.assertEqual(np.isnan(amp).sum(), 0)
			self.assertEqual(np.isnan(ftpow).sum(), 0)

	def test0_filtersb_test_cfreq(self):
		"""Test if filter_sideband() works with varying cfreq"""
		cf = self.cflst[-1]
		fpatt = sim_fringe(np.zeros(self.sz), cf)

		for method in ['spectral', 'passband']:
			for cf in [(-1, -1), (0, 0), (1000, 1000)]:
				phase, amp, ftpow = filter_sideband(fpatt, cf, 0.5, method=method, apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', ret_pow=True, verb=0)
				self.assertEqual(phase.shape, fpatt.shape)
				self.assertEqual(amp.shape, fpatt.shape)
				self.assertEqual(np.isnan(phase).sum(), 0)
				self.assertEqual(np.isnan(amp).sum(), 0)
				self.assertEqual(np.isnan(ftpow).sum(), 0)

	def test0_filtersb_test_sbsize(self):
		"""Test if filter_sideband() works with varying sbsize"""
		cf = self.cflst[-1]
		fpatt = sim_fringe(np.zeros(self.sz), cf)

		for method in ['spectral', 'passband']:
			for sbsz in [0.1, 0.5, 1.0]:
				phase, amp, ftpow = filter_sideband(fpatt, cf, sbsz, method=method, apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', ret_pow=True, verb=0)
				self.assertEqual(phase.shape, fpatt.shape)
				self.assertEqual(amp.shape, fpatt.shape)
				self.assertEqual(np.isnan(phase).sum(), 0)
				self.assertEqual(np.isnan(amp).sum(), 0)
				self.assertEqual(np.isnan(ftpow).sum(), 0)

	def test0_filtersb_test_aptmask(self):
		"""Test if filter_sideband() works with varying aptmask"""
		cf = self.cflst[-1]
		fpatt = sim_fringe(np.zeros(self.sz), cf)
		radmask = tim.im.mk_rad_mask(*fpatt.shape)

		for method in ['spectral', 'passband']:
			for aptrad in [1.0, 0.9, 0.5]:
				aptmask = radmask < aptrad
				phase, amp, ftpow = filter_sideband(fpatt, cf, 0.5, method=method, apt_mask=aptmask, unwrap=True, wsize=-0.5, wfunc='cosine', ret_pow=True, verb=0)
				self.assertEqual(phase.shape, fpatt.shape)
				self.assertEqual(amp.shape, fpatt.shape)
				self.assertEqual(np.isnan(phase).sum(), 0)
				self.assertEqual(np.isnan(amp).sum(), 0)
				self.assertEqual(np.isnan(ftpow).sum(), 0)

	def test1_filtersb_test_zern_sim(self):
		"""Test if filter_sideband() on simulated Zernike wavefront data"""
		### Make fake fringes
		cfreq0 = (18.81111508,  24.76214802)
		zndata = tim.zern.calc_zern_basis(10, min(self.sz)/2)
		rad_mask = tim.im.mk_rad_mask(*zndata['mask'].shape)
		apt_mask = zndata['mask']
		apt_mask9 = rad_mask<0.9

		np.random.seed(1337)
		zvecs = [np.random.random(10) for i in range(6)]
		zphases = [sum(zv*zmode for zv, zmode in zip(zvec, zndata['modes'])) for zvec in zvecs]
		zfringes = [sim_fringe(zph, cfreq0, noiseamp=0) for zph in zphases]

		cfreq = fringe_cal(zfringes, store_pow=False, do_embed=True).mean(0)
		# Should give approximately cfreq, difference should be less 
		# than 2% and pixel difference should be less than 0.3
		self.assertLess(np.abs(1-cfreq/np.r_[cfreq0]).mean(), 0.02)
		self.assertLess(np.abs(cfreq0 - cfreq).mean(), 0.3)

		self.fcache = {}
		for zphase, zfringe in zip(zphases, zfringes):
			phase, amp, ftpow = filter_sideband(zfringe, cfreq0, 0.5, method='spectral', apt_mask=apt_mask, unwrap=True, do_embed=True, wsize=-0.5, wfunc='cosine', ret_pow=True, cache=self.fcache, verb=0)

			dphase = (zphase*apt_mask-phase)
			dphase -= dphase[apt_mask].mean()

			plt.figure(400, figsize=(4,4)); plt.clf()
			ax0 = plt.subplot2grid((2,2),(0, 0))
			ax0.set_title("Input phase")
			im = ax0.imshow(zphase*apt_mask)
			ax1 = plt.subplot2grid((2,2),(0, 1))
			ax1.set_title("Input fringes")
			ax1.imshow(zfringe)
			plt.colorbar(im)
			ax2 = plt.subplot2grid((2,2),(1, 0))
			ax2.set_title("Rec. phase")
			im = ax2.imshow(phase)
			plt.colorbar(im)
			ax3 = plt.subplot2grid((2,2),(1, 1))
			ax3.set_title("In-rec. (inner 90%)")
			im = ax3.imshow(dphase*apt_mask9)
			plt.colorbar(im)

			if (raw_input("Continue [b=break]...") == 'b'): break
		plt.close()

	def test2_filtersb_test_real(self):
		"""Test if filter_sideband() on real data for inspection"""
		# NB: jpgs are required for correct [0,1] data range. PNG data range 
		# [0, 255] gives weird results?
		files = ['fringe_130622_154235Z_000082_img.jpg',
			'fringe_130622_154235Z_000139_img.jpg',
			'fringe_130622_154235Z_000220_img.jpg',
			'fringe_130622_154235Z_000330_img.jpg',
			'fringe_130622_154235Z_000412_img.jpg',
			'fringe_130622_154235Z_000505_img.jpg']
		fringes = [tim.file.read_file(pjoin(TESTDATAPATH,f)) for f in files]

		cfreq = fringe_cal(fringes, store_pow=False, do_embed=True).mean(0)
		apt_mask = tim.im.mk_rad_mask(*fringes[0].shape) < 1

		self.fcache = {}
		phasepow_l = [filter_sideband(i, cfreq, 0.5, method='spectral', apt_mask=apt_mask, unwrap=True, wsize=-0.5, wfunc='cosine', do_embed=True, cache=self.fcache, ret_pow=True, get_complex=False, verb=0) for i in fringes]

		for im, phase in zip(fringes, phasepow_l):
			plt.figure(400, figsize=(4,4)); plt.clf()
			ax0 = plt.subplot2grid((2,2),(0, 0))
			ax0.set_title("fringes")
			ax0.imshow(im)
			ax1 = plt.subplot2grid((2,2),(0, 1))
			ax1.set_title("log side band power")
			ax1.imshow(np.log(phase[2]))
			ax2 = plt.subplot2grid((2,2),(1, 0))
			ax2.set_title("phase")
			ax2.imshow(phase[0])
			ax3 = plt.subplot2grid((2,2),(1, 1))
			ax3.set_title("amplitude")
			ax3.imshow(phase[1])
			if (raw_input("Continue [b=break]...") == 'b'): break
		plt.close()

class TestAvgphase(unittest.TestCase):
	# avg_phase(wavecomps)
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )
		self.cf = (15, 11)
		self.nfr = 32
		self.noiseamp = 2.0
		self.sz = (320, 240)

		minsz = np.min(self.sz)
		apt_mask0 = tim.im.mk_rad_mask(minsz)
		self.apt_mask = np.ones(self.sz)
		self.apt_mask[self.sz[0]/2-minsz/2:self.sz[0]/2+minsz/2, self.sz[1]/2-minsz/2:self.sz[1]/2+minsz/2] = apt_mask0
		self.apt_maskb = self.apt_mask < 1
		self.nan_mask = np.ones_like(self.apt_mask)
		self.nan_mask[self.apt_mask>=1] = np.nan

		np.random.seed(43)
		self.phase0 = 2*np.pi*np.random.random(self.nfr)
		position = np.indices(self.sz)*1./np.r_[self.sz].reshape(-1,1,1)
		self.phase = (3*position**2 + 2*position - 3*position**4).mean(0)
		self.phase -= self.phase.mean()
		self.fringes = [sim_fringe(self.phase, self.cf, noiseamp=np.random.random()*self.noiseamp, phaseoffset=p) for p in self.phase0]

	def test0_plot_input(self):
		"""Show phases and fringes"""
		if (SHOWPLOTS):
			plt.figure(100);plt.clf()
			plt.title("Input phase")
			plt.imshow(self.phase)
			plt.colorbar()

			plt.figure(200, figsize=(4,4));plt.clf()
			plt.subplot(221)
			plt.suptitle("Input fringes")
			plt.imshow(self.fringes[0])
			plt.subplot(222)
			plt.imshow(self.fringes[1])
			plt.subplot(223)
			plt.imshow(self.fringes[2])
			plt.subplot(224)
			plt.imshow(self.fringes[3])
			raw_input("Continue...")
		for i in [100, 200, 201, 202, 203]:
			plt.close(i)

	def test1_test_avg(self):
		"""Compute complex wave components for each fringe, then average"""
		facache = {}
		compl = [filter_sideband(f, self.cf, 0.5, method='spectral', apt_mask=self.apt_maskb, wsize=-0.5, wfunc='cosine', cache=facache, get_complex=True, verb=0) for f in self.fringes]
		vmin, vmax = self.phase.min(), self.phase.max()

		phase, amp = avg_phase(compl)
		phase -= phase.mean()
		dphase = phase - self.phase
		dphase -= dphase.mean()
		dphasev = phase[self.apt_maskb] - self.phase[self.apt_maskb]
		dphasev -= dphasev.mean()
		
		if (SHOWPLOTS):
			plt.figure(100);plt.clf()
			plt.title("Input phase")
			plt.imshow(self.phase, vmin=vmin, vmax=vmax)
			plt.colorbar()

			for i in [0, 5]:
				plt.figure(200+i*10, figsize=(4,2));plt.clf()
				plt.title("Input fringe %d" % i)
				plt.subplot(121)
				plt.imshow(self.fringes[i])
				plt.colorbar()

				ph = np.arctan2(compl[i].imag, compl[i].real)
				ph -= ph.mean()
				plt.subplot(122)
				plt.title("Rec. phase %d" % i)
				plt.imshow(ph, vmin=vmin, vmax=vmax)
				plt.colorbar()

			plt.figure(300, figsize=(4,2));plt.clf()
			plt.subplot(121)
			plt.title("Recovered average phase")
			plt.imshow(phase*self.nan_mask, vmin=vmin, vmax=vmax)
			plt.colorbar()
			plt.subplot(122)
			plt.title("Phase difference")
			plt.imshow(dphase*self.nan_mask, vmin=vmin/10., vmax=vmax/10.)
			plt.colorbar()
			raw_input("Continue...")

		# @todo This assert is probably very dependent on: noise, nfr and 
		# the phase itself
		print "residual RMS/phase RMS", dphasev.std()/phase[self.apt_maskb].std()
		self.assertLess(np.abs(dphasev).std(), vmax/10.)

	def test1_test_avg_weighted(self):
		"""Compute complex wave components for each fringe, then weighted average"""
		compl = [filter_sideband(f, self.cf, 0.5, method='spectral', apt_mask=self.apt_maskb, do_embed=True, wsize=-0.5, wfunc='cosine', get_complex=True, verb=0) for f in self.fringes]
		vmin, vmax = self.phase.min(), self.phase.max()

		phase, amp = avg_phase(compl, ampweight=True)
		phase -= phase.mean()
		dphase = phase - self.phase
		dphase -= dphase.mean()
		dphasev = phase[self.apt_maskb] - self.phase[self.apt_maskb]
		dphasev -= dphasev.mean()

		if (SHOWPLOTS):
			plt.figure(100);plt.clf()
			plt.title("Input phase")
			plt.imshow(self.phase, vmin=vmin, vmax=vmax)
			plt.colorbar()

			for i in [0, 5]:
				plt.figure(200+i*10, figsize=(4,2));plt.clf()
				plt.title("Input fringe %d" % i)
				plt.subplot(121)
				plt.imshow(self.fringes[i])
				plt.colorbar()

				ph = np.arctan2(compl[i].imag, compl[i].real)
				ph -= ph.mean()
				plt.subplot(122)
				plt.title("Rec. phase %d" % i)
				plt.imshow(ph, vmin=vmin, vmax=vmax)
				plt.colorbar()

			plt.figure(300, figsize=(4,2));plt.clf()
			plt.subplot(121)
			plt.title("Recovered average phase")
			plt.imshow(phase*self.nan_mask, vmin=vmin, vmax=vmax)
			plt.colorbar()
			plt.subplot(122)
			plt.title("Phase difference")
			plt.imshow(dphase*self.nan_mask, vmin=vmin/10., vmax=vmax/10.)
			plt.colorbar()
			raw_input("Continue...")

		# @todo This assert is probably very dependent on: noise, nfr and 
		# the phase itself
		print "residual RMS/phase RMS", dphasev.std()/phase[self.apt_maskb].std()
		self.assertLess(np.abs(dphasev).mean(), vmax/10.)

class TestGradphase(unittest.TestCase):
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )

		### Load real fringes
		# NB: jpgs are required for correct [0,1] data range. PNG data range 
		# [0, 255] gives weird results?
		files = ['fringe_130622_154235Z_000082_img.jpg',
			'fringe_130622_154235Z_000139_img.jpg',
			'fringe_130622_154235Z_000220_img.jpg',
			'fringe_130622_154235Z_000330_img.jpg',
			'fringe_130622_154235Z_000412_img.jpg',
			'fringe_130622_154235Z_000505_img.jpg']
		self.fringes = [tim.file.read_file(pjoin(TESTDATAPATH,f)) for f in files]
		self.sz = self.fringes[0].shape
		self.cfreq = (7.81111508,  24.76214802)
		self.fcache = {}
		np.random.seed(1337)

		minsz = np.min(self.sz)
		apt_mask0 = tim.im.mk_rad_mask(minsz)
		# Make circular radial mask in rectangular array
		self.apt_mask = np.ones(self.sz)
		self.apt_mask[self.sz[0]/2-minsz/2:self.sz[0]/2+minsz/2, self.sz[1]/2-minsz/2:self.sz[1]/2+minsz/2] = apt_mask0
		# Make boolean mask and nan mask for plotting
		self.apt_maskb = self.apt_mask < 1
		self.nan_mask = np.ones_like(self.apt_mask)
		self.nan_mask[self.apt_mask>=1] = np.nan

	def test0a_func_calls(self):
		"""Test phase gradient calculation with random data"""
		phase = np.random.random((640, 480))
		phgrad = phase_grad(phase)

		self.assertEqual(len(phgrad), 2)
		self.assertEqual(phgrad[0].shape, phase.shape)
		self.assertEqual(phgrad[1].shape, phase.shape)
		self.assertLess(phgrad[0].ptp(), phase.ptp()*2)
		self.assertLess(phgrad[1].ptp(), phase.ptp()*2)

		phgrad = phase_grad(phase, clip=0.1)
		self.assertLessEqual(phgrad[0].max(), 0.1)
		self.assertLessEqual(phgrad[1].max(), 0.1)
		self.assertGreaterEqual(phgrad[0].min(), -0.1)
		self.assertGreaterEqual(phgrad[1].min(), -0.1)

	def test0b_grad_calc(self):
		"""Test phase gradient calculation from Zernike phases"""
		zndata = tim.zern.calc_zern_basis(10, min(self.sz)/2)
		
		zvecs = [np.random.random(10) for i in range(6)]
		zphases = [sum(zv*zmode for zv, zmode in zip(zvec, zndata['modes'])) for zvec in zvecs]

		for phase in zphases:
			phgrad = phase_grad(phase)
			self.assertEqual(len(phgrad), 2)
			self.assertEqual(phgrad[0].shape, phase.shape)
			self.assertEqual(phgrad[1].shape, phase.shape)
			self.assertLess(phgrad[0].ptp(), phase.ptp()*2)
			self.assertLess(phgrad[1].ptp(), phase.ptp()*2)

		if (SHOWPLOTS):
			plt.figure(200, figsize=(4,4));plt.clf()
			plt.subplot(221)
			plt.title("Input phase")
			plt.imshow(phase)
			plt.colorbar()

			plt.subplot(222)
			plt.title("Sum(grad)")
			plt.imshow(phgrad[0]+phgrad[1])
			plt.colorbar()

			plt.subplot(223)
			plt.title("Grad 0")
			plt.imshow(phgrad[0])
			plt.colorbar()

			plt.subplot(224)
			plt.title("Grad 1")
			plt.imshow(phgrad[1])
			plt.colorbar()
			raw_input("Continue...")

	def test1a_zern_rec(self):
		"""Test phase gradient calculation and recovery from Zernike phases"""
		zndata = tim.zern.calc_zern_basis(10, min(self.sz)/2)
		
		zvecs = [np.random.random(10) for i in range(6)]
		zphases = [sum(zv*zmode for zv, zmode in zip(zvec, zndata['modes'])) for zvec in zvecs]
		zngradmat = np.r_[ [phase_grad(zmode, asvec=True) for zmode in zndata['modes']] ]

		for zphase, zvec in zip(zphases, zvecs):
			zphgradvec = phase_grad(zphase, asvec=True)
			zphgrad = phase_grad(zphase)

			# Now fit Zernikes on gradients
			# Fit both gradients simultaneously
			# This works best as modes with no [x,y] gradient still have 
			# [y,x] gradient
			zngrad_vec = np.dot(zphgradvec, np.linalg.pinv(zngradmat)).ravel()

			if (SHOWPLOTS):
				plt.figure(200, figsize=(4,4));plt.clf()
				plt.subplot(221)
				plt.title("Input phase")
				plt.imshow(zphase*self.nan_mask)
				plt.colorbar()

				plt.subplot(222)
				plt.title("Sum(grad)")
				plt.imshow(zphgrad[0]+zphgrad[1])
				plt.colorbar()

				plt.subplot(223)
				plt.title("Zvec vectors")
				plt.plot(zvec, 'o-', label='Input')
				plt.plot(zngrad_vec, '^--', label='Grad. rec.')
				plt.legend(loc='best')

				plt.subplot(224)
				plt.title("Zvec diff")
				plt.plot(zvec[1:]-zngrad_vec[1:], 'o')
				if (raw_input("Continue [b=break]...") == 'b'): break

			np.testing.assert_allclose(zvec[1:], zngrad_vec[1:])

	def test1b_zern_fringe_rec(self):
		"""Test phase gradient calculation and recovery from Zernike fringes"""
		zndata = tim.zern.calc_zern_basis(10, min(self.sz)/2)
		
		# Generate phases from Zernike modes
		zvecs = [np.random.random(10) for i in range(6)]
		zphases = [sum(zv*zmode for zv, zmode in zip(zvec, zndata['modes'])) for zvec in zvecs]
		zmodemat = np.r_[ [zm[self.apt_maskb] for zm in zndata['modes']] ]
		# Calculate gradient from Zernike phase
		zngradmat = np.r_[ [phase_grad(zmode, apt_mask=self.apt_maskb, asvec=True) for zmode in zndata['modes']] ]

		# Simulate fringes
		zfringes = [sim_fringe(zph, self.cfreq, noiseamp=0) for zph in zphases]

		cfreq = fringe_cal(zfringes, store_pow=False, do_embed=True).mean(0)
		# Should give approximately cfreq, difference should be less 
		# than 2% and pixel difference should be less than 0.3
		self.assertLess(np.abs(1-cfreq/np.r_[self.cfreq]).mean(), 0.02)
		self.assertLess(np.abs(self.cfreq - cfreq).mean(), 0.3)

		zvecs_reg = []
		zvecs_grad = []

		# Now recover phase from fringes through gradients
		for zfringe, zphase, zvec in zip(zfringes, zphases, zvecs):
			phase, amp, ftpow = filter_sideband(zfringe, cfreq, 0.5, method='spectral', apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', ret_pow=True, cache=self.fcache, verb=0)
			vmin, vmax = phase.min(), phase.max()
			phgrad = phase_grad(phase)

			plt.figure(200, figsize=(4,4));plt.clf()
			plt.subplot(221)
			plt.title("Input phase")
			plt.imshow(zphase*self.nan_mask)
			plt.colorbar()

			plt.subplot(222)
			plt.title("Recovered phase")
			plt.imshow(phase*self.nan_mask)
			plt.colorbar()

			plt.subplot(223)
			plt.title("Phase grad sum")
			plt.imshow((phgrad[0]+phgrad[1])*self.nan_mask)
			plt.colorbar()

			# From recovered phase and gradients, fit Zernikes
			znrec_vec = np.dot(phase[self.apt_maskb], np.linalg.pinv(zmodemat)).ravel()
			zvecs_reg.append(znrec_vec)

			# Now fit Zernikes on gradients
			# Fit both gradients simultaneously. This works best as modes 
			# with no [x,y] gradient still have [y,x] gradient
			zphgradvec = phase_grad(phase, apt_mask=self.apt_maskb, asvec=True)
			zngrad_vec = np.dot(zphgradvec,np.linalg.pinv(zngradmat)).ravel()
			zvecs_grad.append(zngrad_vec)

			plt.subplot(224)
			plt.title("Zernike vectors")
			nz = np.arange(10)+1
			plt.plot(nz[1:]-0.1, zvec[1:], 'o-')
			plt.plot(nz[1:]-0.0, znrec_vec[1:], 's--', label="Reg. rec.")
			plt.plot(nz[1:]+0.1, zngrad_vec[1:], '^--', label="Grad. rec.")
			plt.xlabel("Zernike mode [Noll]")
			plt.ylabel("Amplitude [unit rms]")
			plt.legend()

			if (raw_input("Continue [b=break]...") == 'b'): break

	def test2a_data_fringe_rec(self):
		"""Test phase gradient calculation on data fringes"""
		cfreq = fringe_cal(self.fringes, store_pow=False, do_embed=True).mean(0)
		zndata = tim.zern.calc_zern_basis(10, min(self.sz)/2)
		zmodemat = np.r_[ [zm[self.apt_maskb] for zm in zndata['modes']] ]
		zngradmat = np.r_[ [phase_grad(zmode, apt_mask=self.apt_maskb, asvec=True) for zmode in zndata['modes']] ]

		zvecs_reg = []
		zvecs_regw = []
		zvecs_grad = []

		for fringe in self.fringes:
			phase, amp, ftpow = filter_sideband(fringe, cfreq, 0.5, method='spectral', apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', ret_pow=True, verb=0)

			# Fit Zernike on scalar values
			znrec_vec = np.dot(phase[self.apt_maskb], np.linalg.pinv(zmodemat)).ravel()
			zvecs_reg.append(znrec_vec)

			# Fit Zernikes on gradients
			phgradvec = phase_grad(phase, clip=0.2, apt_mask=self.apt_maskb, asvec=True)
			zngrad_vec = np.dot(phgradvec, np.linalg.pinv(zngradmat)).ravel()
			zvecs_grad.append(zngrad_vec)

			# Fit Zernike on scalar values, weighted
			zvec, zrec, zdiff = tim.zern.fit_zernike(phase, zern_data=zndata, fitweight=amp)
			zvecs_regw.append(zvec)

		plt.figure(400); plt.clf()
		plt.title("Zernike vectors for 6 fringes")
		nz = np.arange(10)+1
		plt.errorbar(nz-0.1, np.mean(zvecs_reg,0), yerr=np.std(zvecs_reg,0), elinewidth=3, fmt='.', label='Reg.')
		plt.errorbar(nz+0.0, np.mean(zvecs_grad,0), yerr=np.std(zvecs_grad,0), elinewidth=3, fmt='.', label='Grad.')
		plt.errorbar(nz+0.1, np.mean(zvecs_regw,0), yerr=np.std(zvecs_regw,0), elinewidth=3, fmt='.', label='Reg. wt.')
		plt.legend(loc='best')

		tim.shell()

class TestCalcPhaseVec(unittest.TestCase):
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )

		### Load real fringes
		# NB: jpgs are required for correct [0,1] data range. PNG data range 
		# [0, 255] gives weird results?
		files = ['fringe_130622_154235Z_000082_img.jpg',
			'fringe_130622_154235Z_000139_img.jpg',
			'fringe_130622_154235Z_000220_img.jpg',
			'fringe_130622_154235Z_000330_img.jpg',
			'fringe_130622_154235Z_000412_img.jpg',
			'fringe_130622_154235Z_000505_img.jpg']
		self.fringes = [tim.file.read_file(pjoin(TESTDATAPATH,f)) for f in files]
		self.sz = self.fringes[0].shape
		self.cfreq = (7.81111508,  24.76214802)
		self.fcache = {}
		np.random.seed(1337)

		minsz = np.min(self.sz)
		apt_mask0 = tim.im.mk_rad_mask(minsz)
		# Make circular radial mask in rectangular array
		self.apt_mask = np.ones(self.sz)
		self.apt_mask[self.sz[0]/2-minsz/2:self.sz[0]/2+minsz/2, self.sz[1]/2-minsz/2:self.sz[1]/2+minsz/2] = apt_mask0
		self.apt_mask = self.apt_mask < 1
		# Make NaN aperture mask for plotting
		self.nan_mask = np.ones(self.sz)
		self.nan_mask[self.apt_mask == False] = np.nan

	def test0_calls_scalar(self):
		"Test if calling the function with method=scalar works"
		rnd = np.random.random
		fakewaves = [rnd(self.sz) + rnd(self.sz)*1j for i in range(10)]
		fakemat = rnd(self.sz + (10,))

		calc_phasevec(fakewaves, fakemat.reshape(-1,10), method='scalar')
		calc_phasevec(fakewaves, fakemat.reshape(-1,10), method='scalar', apt_mask=self.apt_mask)
		calc_phasevec(fakewaves, fakemat.reshape(-1,10)[:,:1], method='scalar', apt_mask=self.apt_mask)
		calc_phasevec([w[::2,::2] for w in fakewaves] , fakemat[::2,::2].reshape(-1,10), method='scalar', apt_mask=self.apt_mask[::2,::2])

	def test0_calls_gradient(self):
		"Test if calling the function with method=gradient works"
		rnd = np.random.random
		fakewaves = [rnd(self.sz) + rnd(self.sz)*1j for i in range(10)]
		fakemat = rnd((np.product(self.sz), 10))
		thiscache = {}

		calc_phasevec(fakewaves, fakemat, method='gradient')
		calc_phasevec(fakewaves, fakemat, method='gradient', apt_mask=self.apt_mask)
		calc_phasevec(fakewaves, fakemat[:,:1], method='gradient', apt_mask=self.apt_mask, cache=thiscache)
		calc_phasevec(fakewaves, fakemat[:,:1], method='gradient', apt_mask=self.apt_mask, cache=thiscache)
		calc_phasevec([w[::2,::2] for w in fakewaves] , fakemat[::4], method='gradient', apt_mask=self.apt_mask[::2,::2])

	def test0_calls_vshwfs(self):
		"Test if calling the function with method=vshwfs works"
		# Make fake microlens grid
		x0arr = np.arange(self.sz[0]*0.25, (self.sz[0]-16)*0.75, 16)
		y0arr = np.arange(self.sz[1]*0.10, (self.sz[1]-16)*0.90, 16)
		mlagrid = [(x0, x0+16, y0, y0+16) for x0 in x0arr for y0 in y0arr]

		rnd = np.random.random
		fakewaves = [rnd(self.sz) + rnd(self.sz)*1j for i in range(10)]
		fakemat = rnd((np.product(self.sz), 10))
		thiscache = {}

		calc_phasevec(fakewaves, fakemat, method='vshwfs', mlagrid=mlagrid)
		calc_phasevec(fakewaves, fakemat, method='vshwfs', mlagrid=mlagrid, scale=3)
		calc_phasevec(fakewaves, fakemat, method='vshwfs', mlagrid=mlagrid, scale=3, cache=thiscache)
		calc_phasevec(fakewaves, fakemat, method='vshwfs', mlagrid=mlagrid, scale=3, cache=thiscache)
		calc_phasevec(fakewaves, fakemat[:,:1], method='vshwfs', mlagrid=mlagrid, scale=3)

	def test2_consistency(self, shscl=2):
		"Test if the three methods give consistent results"
		# Compute Zernike basis set 
		zern_data = tim.zern.calc_zern_basis(11, self.apt_mask.shape[0]/2, modestart=2, calc_covmat=False)

		# Make fake microlens grid
		x0arr = np.arange(self.sz[0]*0.25, (self.sz[0]-16)*0.75, 16)
		y0arr = np.arange(self.sz[1]*0.10, (self.sz[1]-16)*0.90, 16)
		mlagrid = [(x0, x0+16, y0, y0+16) for x0 in x0arr for y0 in y0arr]

		# Compute basis mode matrices for different methods
		zmodemat = zern_data['modesmat'].T

		# Make fake complex waves for a random Zernike phase
		zvec = np.random.random(11)-0.5
		zphase = np.dot(zern_data['modesmat'].T, zvec).reshape(zern_data['modes'][0].shape)
		waves = [np.exp(1j*zphase)]

		# Recover Zernike vector from complex wave
		zvec0, zimg0 = calc_phasevec(waves, zmodemat, method='scalar', apt_mask=self.apt_mask)
		zvec1, zimg1 = calc_phasevec(waves, zmodemat, method='gradient', apt_mask=self.apt_mask)
		zvec2, zimg2 = calc_phasevec(waves, zmodemat, method='vshwfs', apt_mask=self.apt_mask, mlagrid=mlagrid, scale=shscl)

		self.assertAlmostEqual(np.abs(zvec0-zvec).mean(), 0, delta=0.01)
		np.testing.assert_allclose(zvec0, zvec, atol=0.05)

	def test2_speedtest(self):
		"Test speed for different reconstruction methods"
		t1 = Timer("""
a=calc_phasevec(fakewaves, fakemat, method='scalar')
		""", """import numpy as np
from fringe import calc_phasevec
sz = (256, 256)
rnd = np.random.random
fakewaves = [rnd(sz) + rnd(sz)*1j for i in range(4)]
fakemat = rnd((np.product(sz), 20))""")

		t2 = Timer("""
a=calc_phasevec(fakewaves, fakemat, method='scalar')
		""", """import numpy as np
from fringe import calc_phasevec
sz = (256, 256)
rnd = np.random.random
fakewaves = [rnd(sz) + rnd(sz)*1j for i in range(4)]
fakemat = rnd((np.product(sz)*2, 20))""")

		t3 = Timer("""
a=calc_phasevec(fakewaves, fakemat, method='vshwfs', mlagrid=mlagrid, scale=2)
		""", """import numpy as np
from fringe import calc_phasevec
sz = (256, 256)
sasz = 16
x0arr = np.arange(sz[0]*0.25, (sz[0]-sasz)*0.75, sasz).astype(int)
y0arr = np.arange(sz[1]*0.11, (sz[1]-sasz)*0.90, sasz).astype(int)
mlagrid = [(x0, x0+sasz, y0, y0+sasz) for x0 in x0arr for y0 in y0arr]

rnd = np.random.random
fakewaves = [rnd(sz) + rnd(sz)*1j for i in range(4)]
fakemat = rnd((len(mlagrid)*2, 20))""")

		print "calc_phasevec(): timing results:"
		t_scalar = 1e3*min(t1.repeat(2, 10))/10
		print "calc_phasevec(): scalar %.3g msec/it" % (t_scalar)
		t_gradient = 1e3*min(t2.repeat(2, 10))/10
		print "calc_phasevec(): gradient %.3g msec/it" % (t_gradient)
		t_vshwfs = 1e3*min(t3.repeat(2, 10))/10
		print "calc_phasevec(): vshwfs %.3g msec/it" % (t_vshwfs)



if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())

### EOF

