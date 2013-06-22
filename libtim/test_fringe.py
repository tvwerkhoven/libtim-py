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

# Import libs
from fringe import *
import libtim as tim
import unittest
import numpy as np
import pylab as plt
import logging
logging.basicConfig( stream=sys.stderr )
logging.getLogger( "test_fringe" ).setLevel( logging.DEBUG )

SHOWPLOTS=True

class TestFringecal(unittest.TestCase):
	# fringe_cal(refimgs, wsize=-0.5, cpeak=0, do_embed=True, store_pow=True, ret_pow=False, outdir='./'):
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )
		self.cflst = [(3, 4), (18.3, 1.3), (22.22, 11.11)]
		self.sz = (640, 480)

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
			cfreq1 = fringe_cal([fpatt], store_pow=False, do_embed=False)[0]
			cfrat = 1.-np.r_[cfreq]/cfreq1
			self.log.debug("ref: %.3f, %.3f, rec: %.3f, %.3f, ratio: %.2g, %.2g", cfreq[0], cfreq[1], cfreq1[0], cfreq1[1], cfrat[0], cfrat[1])
			self.assertAlmostEqual(sum(cfreq1), sum(cfreq), places=0)
			np.testing.assert_almost_equal(cfreq1, cfreq, decimal=0)

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

	def test3_cal_qual(self):
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
				self.log.debug("cf: %d, noise: %#5.3g, ratio1*e3: %+#4.2f, %+#4.2f, ratio2*e3: %+#4.2f, %+#4.2f", idx, noiseamp, cfrat1[0], cfrat1[1], cfrat2[0], cfrat2[1])

class TestFiltersb(unittest.TestCase):
	# filter_sideband(img, cfreq, sbsize, method='spectral', apt_mask=None, unwrap=True, wsize=-0.5, wfunc='cosine', cache={}, ret_pow=False, verb=0)
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )
		self.cflst = [(3, 4), (18.3, 1.3), (22.22, 11.11)]
		self.sz = (640, 480)

	def test0_filtersb_test_nom(self):
		"""Test if filter_sideband() works nominally"""
		cf = self.cflst[-1]
		fpatt = sim_fringe(np.zeros(self.sz), cf)

		# Test nominal
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

		# Test carrfreq
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

		# Test sbsize
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

		# Test sbsize
		for method in ['spectral', 'passband']:
			for aptrad in [1.0, 0.9, 0.5]:
				aptmask = radmask < aptrad
				phase, amp, ftpow = filter_sideband(fpatt, cf, 0.5, method=method, apt_mask=aptmask, unwrap=True, wsize=-0.5, wfunc='cosine', ret_pow=True, verb=0)
				self.assertEqual(phase.shape, fpatt.shape)
				self.assertEqual(amp.shape, fpatt.shape)
				self.assertEqual(np.isnan(phase).sum(), 0)
				self.assertEqual(np.isnan(amp).sum(), 0)
				self.assertEqual(np.isnan(ftpow).sum(), 0)


class TestAvgphase(unittest.TestCase):
	# avg_phase(wavecomps)
	def setUp(self):
		self.log = logging.getLogger( "test_fringe" )
		self.cf = (5, 7)
		self.nfr = 32
		self.noiseamp = 2.0
		self.sz = (640, 480)

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
			plt.imshow(self.phase)
			plt.colorbar()

			plt.figure(200);plt.clf()
			plt.imshow(self.fringes[0])
			plt.colorbar()

			plt.figure(201);plt.clf()
			plt.imshow(self.fringes[1])
			plt.colorbar()

			plt.figure(202);plt.clf()
			plt.imshow(self.fringes[2])
			plt.colorbar()

			plt.figure(203);plt.clf()
			plt.imshow(self.fringes[3])
			plt.colorbar()
			raw_input("...")

	def test1_test_avg(self):
		"""Compute complex wave components for each fringe, then average"""
		compl = [filter_sideband(f, self.cf, 0.5, method='spectral', apt_mask=None, wsize=-0.5, wfunc='cosine', get_complex=True, verb=0) for f in self.fringes]
		vmin, vmax = self.phase.min(), self.phase.max()

		phase, amp = avg_phase(compl)
		phase -= phase.mean()
		dphase = phase - self.phase
		dphase -= dphase.mean()
		
		if (SHOWPLOTS):
			plt.figure(100);plt.clf()
			plt.title("Input phase")
			plt.imshow(self.phase, vmin=vmin, vmax=vmax)
			plt.colorbar()

			for i in [0, 5]:
				plt.figure(110+i*10);plt.clf()
				plt.title("Input fringe %d" % i)
				plt.imshow(self.fringes[i])
				plt.colorbar()

				ph = np.arctan2(compl[i].imag, compl[i].real)
				ph -= ph.mean()
				plt.figure(111+i*10);plt.clf()
				plt.title("Rec. phase %d" % i)
				plt.imshow(ph, vmin=vmin, vmax=vmax)
				plt.colorbar()

			plt.figure(200);plt.clf()
			plt.title("Recovered phase")
			plt.imshow(phase, vmin=vmin, vmax=vmax)
			plt.colorbar()

			plt.figure(300);plt.clf()
			plt.title("Phase difference")
			plt.imshow(dphase, vmin=vmin/10., vmax=vmax/10.)
			plt.colorbar()
			raw_input("...")

		# @todo This assert is probably very dependent on: noise, nfr and 
		# the phase itself
		print "mean(abs(phase residual)):", np.abs(dphase).mean()
		print "max(phase)/10:", vmax/10.
		self.assertLess(np.abs(dphase).mean(), vmax/10.)

	def test1_test_avg_weighted(self):
		"""Compute complex wave components for each fringe, then weighted average"""
		compl = [filter_sideband(f, self.cf, 0.5, method='spectral', apt_mask=None, wsize=-0.5, wfunc='cosine', get_complex=True, verb=0) for f in self.fringes]
		vmin, vmax = self.phase.min(), self.phase.max()

		phase, amp = avg_phase(compl, ampweight=True)
		phase -= phase.mean()
		dphase = phase - self.phase
		dphase -= dphase.mean()

		# @todo This assert is probably very dependent on: noise, nfr and 
		# the phase itself
		print "mean(abs(phase residual)):", np.abs(dphase).mean()
		print "max(phase)/10:", vmax/10.
		self.assertLess(np.abs(dphase).mean(), vmax/10.)

if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())

### EOF

