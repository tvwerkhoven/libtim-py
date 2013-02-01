#!/usr/bin/env python
# encoding: utf-8
"""
@file test_lightcurve.py
@brief Test libtim/lightcurve.py library

Get profiling results using
nosetests-2.7 --with-profile ./test_lightcurve.py

@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20121029

Testcases for lightcurve.py library.
"""

from lightcurve import *
import unittest
import pylab as plt
import pyfits
from timeit import Timer
import os

SHOWPLOTS=False

def plotfunc(testdat, refdat, fignum, pause=True):
	if (not SHOWPLOTS): return
	plt.figure(fignum)
	plt.clf()
	plt.plot(testdat, label='test')
	plt.plot(refdat, label='reference')
	plt.plot(testdat/refdat, label='ratio')
	plt.legend()
	if (pause):
		raw_input('')

class ModelCrossCheck(unittest.TestCase):
	def test0_sanity(self):
		"""Check step 0: function should work"""
		dp7_tim = transit_model_dp7([0])

	def test1_comp_cloud(self):
		"""Check step 1: cloud curve"""
		dp7_tim = transit_model_dp7([0], sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=1, verb=0, plot=0)
		dp7_matteo = pyfits.getdata('test_lightcurve_in/01_cloud.fits')

		plotfunc(dp7_tim, dp7_matteo, 1)
		self.assertAlmostEqual(np.abs(dp7_tim - dp7_matteo).max(), 0.0)
		self.assertTrue(np.allclose(dp7_tim, dp7_matteo))

	def test2_comp_star(self):
		"""Check step 2: star curve"""
		dp7_tim = transit_model_dp7([0], sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=2, verb=0, plot=0)
		dp7_matteo = pyfits.getdata('test_lightcurve_in/02_star.fits')

		plotfunc(dp7_tim, dp7_matteo, 2)
		self.assertAlmostEqual(np.abs(dp7_tim - dp7_matteo).max(), 0.0)
		self.assertTrue(np.allclose(dp7_tim, dp7_matteo))

	def test3_comp_lc0(self):
		"""Check step 3: check cloud convolved with star"""
		dp7_tim = transit_model_dp7([0], sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=3, verb=0, plot=0)
		dp7_matteo = pyfits.getdata('test_lightcurve_in/03_lc_abs.fits')

		plotfunc(dp7_tim, dp7_matteo, 3)
		self.assertAlmostEqual(np.abs(dp7_tim - dp7_matteo).max(), 0.0)
		self.assertTrue(np.allclose(dp7_tim, dp7_matteo))

	def test4_comp_star(self):
		"""Check step 4: scattering curve"""
		dp7_tim = transit_model_dp7([0], sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=4, verb=0, plot=0)
		dp7_matteo = pyfits.getdata('test_lightcurve_in/04_scatt.fits')

		plotfunc(dp7_tim, dp7_matteo, 4)
		self.assertAlmostEqual(np.abs(dp7_tim - dp7_matteo).max(), 0.0, places=3)
		# self.assertTrue(np.allclose(dp7_tim, dp7_matteo))


	def test5_comp_lc1(self):
		"""Check step 5: light curve + scattering"""
		dp7_tim = transit_model_dp7([0], sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=5, verb=0, plot=0)
		dp7_matteo = pyfits.getdata('test_lightcurve_in/05_lc_abs+scatt.fits')

		plotfunc(dp7_tim, dp7_matteo, 5)
		self.assertAlmostEqual(np.abs(dp7_tim - dp7_matteo).max(), 0.0, places=4)
		# self.assertTrue(np.allclose(dp7_tim, dp7_matteo))

	def test51_comp_lc1(self):
		"""Check step 51: light curve - scattering"""
		dp7_tim = transit_model_dp7([0], sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=51, verb=0, plot=0)
		dp7_matteo_lc1 = pyfits.getdata('test_lightcurve_in/05_lc_abs+scatt.fits')
		dp7_matteo_lc0 = pyfits.getdata('test_lightcurve_in/03_lc_abs.fits')
		dp7_matteo_lc_scatt = dp7_matteo_lc1-dp7_matteo_lc0

		plotfunc(dp7_tim, dp7_matteo_lc_scatt, 51)
		self.assertAlmostEqual(np.abs(dp7_tim - dp7_matteo_lc_scatt).max(), 0.0, places=4)
		# self.assertTrue(np.allclose(dp7_tim, dp7_matteo_lc_scatt))

	def test6_comp_lc2(self):
		"""Check step 6: final convolved light curve"""
		dp7_tim = transit_model_dp7([0], sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=6, verb=0, plot=0)
		dp7_matteo = pyfits.getdata('test_lightcurve_in/06_lc_kep_conv.fits')

		plotfunc(dp7_tim, dp7_matteo, 6)
		self.assertAlmostEqual(np.abs(dp7_tim - dp7_matteo).max(), 0.0, places=4)
		# self.assertTrue(np.allclose(dp7_tim, dp7_matteo))

class ModelSpeed(unittest.TestCase):
	def test1_speed(self):
		"""Measure model speed"""
		t1 = Timer("""
dp7_tim = transit_model_dp7(ph, sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=0, verb=0, plot=0)
		""", """
from lightcurve import transit_model_dp7
import numpy as np
ph = 2.0*np.pi*np.arange(1000)/1000.0
		""")

		t1_min = min(t1.repeat(3, 100))/100
		print u"test1_speed(): transit_model_dp7 %.3g ms/it" % (t1_min*1000)

def plotcomp(lclist, fignum, title, pause=True):
	"""
	Plot list of light curves to compare
	"""
	if (not SHOWPLOTS): return

	plt.figure(fignum)
	plt.clf()
	plt.title(title)
	for lc in lclist:
		plt.plot(lc)
	if (pause):
		raw_input('')

class ModelParamsPlot(unittest.TestCase):
	# transit_model_dp7(ph, sr=10.36, ep=5.13, ca=0.03, g=0.875, om=0.654, nmodel=400, method=0, verb=0, plot=0)
	def test1_param_sr(self):
		"""Test sr parameter"""
		lclist = [transit_model_dp7([], sr=pfac*10.36) for pfac in np.linspace(0.5, 2.0, 10)]
		plotcomp(lclist, 1, 'sr 0.5..2.0*10.36')
	def test2_param_ep(self):
		"""Test ep parameter"""
		lclist = [transit_model_dp7([], ep=pfac*5.13) for pfac in np.linspace(0.5, 2.0, 10)]
		plotcomp(lclist, 2, 'ep 0.5...2.0*5.13')
	def test3_param_ca(self):
		"""Test ca parameter"""
		lclist = [transit_model_dp7([], ca=pfac*0.03) for pfac in np.linspace(0.5, 2.0, 10)]
		plotcomp(lclist, 3, 'ca 0.5...2.0*0.03')
	def test4_param_g(self):
		"""Test g parameter"""
		lclist = [transit_model_dp7([], g=pfac*0.875) for pfac in np.linspace(0.5, 2.0, 10)]
		plotcomp(lclist, 4, 'g 0.5...2.0*0.875')
	def test5_param_om(self):
		"""Test om parameter"""
		lclist = [transit_model_dp7([], om=pfac*0.654) for pfac in np.linspace(0.5, 2.0, 10)]
		plotcomp(lclist, 5, 'om 0.5...2.0*0.654')
	def test6_param_nmodel(self):
		"""Test nmodel parameter"""
		lclist = [transit_model_dp7([], nmodel=int(pfac*400)) for pfac in np.linspace(0.5, 2.0, 10)]
		plotcomp(lclist, 6, 'nmodel 0.5...2.0*400')

if __name__ == "__main__":
        import sys
        sys.exit(unittest.main())

# EOF
