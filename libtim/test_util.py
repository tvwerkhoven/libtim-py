#!/usr/bin/env python
# encoding: utf-8
"""
@file test_util.py
@brief Test suite for util.py
"""

#============================================================================
# Import libraries here
#============================================================================

import unittest
from util import *

#============================================================================
# Unit tests
#============================================================================

class TestUptime(unittest.TestCase):
	def setUp(self):
		self.samples = {}
		self.samples['OSX'] = [
			"21:52  up 6 days,  4:49, 4 users, load averages: 0.26 0.32 0.38",
			"18:30 up 1 min, 2 users, load averages: 0.71 0.20 0.07",
			"14:00 up 36 mins, 2 users, load averages: 0.82 0.51 0.33",
			"12:45 up 23:21, 4 users, load averages: 4.51 5.34 5.12",
			"13:30 up 1 day, 6 mins, 4 users, load averages: 0.71 0.60 1.02",
			"1:57 up 2 days, 12:19, 5 users, load averages: 0.56 0.51 0.61",
			"14:30 up 1 day, 1:05, 2 users, load averages: 0.55 0.48 0.55"]
		
		self.samples['Debian'] = [
			"20:45:11 up 0 min,  1 user,  load average: 0.55, 0.17, 0.06",
			"20:45:17 up 1 min,  1 user,  load average: 0.51, 0.17, 0.06",
			"20:51:39 up 7 min,  1 user,  load average: 0.50, 0.16, 0.06",
			"20:55:56 up 11 min,  1 user,  load average: 0.02, 0.08, 0.05",
			"22:23:27 up  1:39,  1 user,  load average: 0.00, 0.00, 0.00"]

		self.samples['NetBSD'] = [
			"9:55AM up 1 min, 1 user, load averages: 0.11, 0.12, 0.14",
			"10:55AM up 1 hr, 1 user, load averages: 0.11, 0.12, 0.14",
			"12:00PM up 6 days, 14:42, 1 user, load averages: 0.25, 0.17, 0.10",
			"10:26AM up 32 mins, 1 user, load averages: 0.11, 0.12, 0.14",
			"10:56AM up 1:03, 1 user, load averages: 0.25, 0.24, 0.18",
			"10:56AM up 6 days, 32 mins, 1 user, load averages: 0.25, 0.24, 0.18",
			"8:53AM up 23 hrs, 1 user, load averages: 0.13, 0.13, 0.09",
			"9:54AM up 1 day, 1 user, load averages: 0.34, 0.29, 0.16",
			"9:55AM up 1 day, 1 min, 1 user, load averages: 0.44, 0.31, 0.17",
			"10:55AM up 1 day, 1 hr, 1 user, load averages: 0.44, 0.31, 0.17"]

		self.resp = {}
		self.resp['OSX'] = [
			('21:52', 6.200694444444445, 4, (0.26, 0.32, 0.38)),
			('18:30', 0.0006944444444444445, 2, (0.71, 0.2, 0.07)),
			('14:00', 0.025, 2, (0.82, 0.51, 0.33)),
			('12:45', 0.9729166666666667, 4, (4.51, 5.34, 5.12)),
			('13:30', 1.0041666666666667, 4, (0.71, 0.6, 1.02)),
			('1:57', 2.5131944444444443, 5, (0.56, 0.51, 0.61)),
			('14:30', 1.045138888888889, 2, (0.55, 0.48, 0.55))]
		self.resp['Debian'] = [
			('20:45:11', 0.0, 1, (0.55, 0.17, 0.06)),
			('20:45:17', 0.0006944444444444445, 1, (0.51, 0.17, 0.06)),
			('20:51:39', 0.004861111111111111, 1, (0.5, 0.16, 0.06)),
			('20:55:56', 0.007638888888888889, 1, (0.02, 0.08, 0.05)),
			('22:23:27', 0.06875, 1, (0.0, 0.0, 0.0))]
		self.resp['NetBSD'] = [
			('9:55AM', 0.0006944444444444445, 1, (0.11, 0.12, 0.14)),
			('10:55AM', 0.041666666666666664, 1, (0.11, 0.12, 0.14)),
			('12:00PM', 6.6125, 1, (0.25, 0.17, 0.1)),
			('10:26AM', 0.022222222222222223, 1, (0.11, 0.12, 0.14)),
			('10:56AM', 0.04375, 1, (0.25, 0.24, 0.18)),
			('10:56AM', 6.022222222222222, 1, (0.25, 0.24, 0.18)),
			('8:53AM', 0.9583333333333334, 1, (0.13, 0.13, 0.09)),
			('9:54AM', 1.0, 1, (0.34, 0.29, 0.16)),
			('9:55AM', 1.0006944444444446, 1, (0.44, 0.31, 0.17)),
			('10:55AM', 1.0416666666666667, 1, (0.44, 0.31, 0.17))]

	def test0_call_ok(self):
		"""Call function, check for failure"""
		parse_uptime(self.samples['OSX'][0])

	def test0_call_bad(self):
		"""Call function with empty string, confirm failure"""
		with self.assertRaises(ValueError):
			parse_uptime("")

	def test0_call_ret_var(self):
		"""Call function, check number of args ok"""
		t, u, n, l = parse_uptime(self.samples['OSX'][0])

	def test1_osx_syntax(self):
		"""Test function with OSX input data"""
		for instr in self.samples['OSX']:
			parsed = parse_uptime(instr)
			t, u, n, l = parsed

	def test2_parse_all(self):
		"""Test function with all input data"""
		for key, data in self.samples.iteritems():
			for instr in data:
				parsed = parse_uptime(instr)
				t, u, n, l = parsed

	def test3_confirm_all(self):
		"""Test function with all input data"""
		for key in self.samples.keys():
			for instr, resp in zip(self.samples[key], self.resp[key]):
				parsed = parse_uptime(instr)
				t, u, n, l = parsed
				self.assertEqual(parsed, resp, msg="Failed for %s data '%s'" % (key, instr))
				

class TestGitRev(unittest.TestCase):
	def setUp(self):
		pass

	def test0_call(self):
		"""Call function, check for failure"""
		# Current directory probably has a revision
		git_rev('./')
		# Empty string defaults to current dir
		git_rev('')
		# 'no file' defaults to current directory
		git_rev('no file')

		# /tmp/ probably has no git revision
		self.assertEqual(git_rev('/tmp/'), '')
		self.assertRaises(TypeError, git_rev, 1)

	def test1_rev_lib(self):
		"""
		Query revision of this lib
		@todo This dir is not always a git repo! How to test?
		"""
		rev = git_rev(sys.argv[0])
		#print rev
		self.assertTrue(rev.__class__ == 'string'.__class__)
		self.assertTrue(len(rev) >= 0)
		self.assertTrue(len(rev) < 32)

class TestMetaData(unittest.TestCase):
	def setUp(self):
		self.meta = {'hello': 'world'}
		self.outfiles = {}

	def tearDown(self):
		"""Delete testfiles if necessary"""
		for format, file in self.outfiles.iteritems():
			if (file and os.path.isfile(file)):
				os.remove(file)

	def test1a_gen_meta_plain(self):
		"""Test metadata generation"""
		data = gen_metadata(self.meta)

	def test1b_gen_meta_args(self):
		"""Test metadata generation with *args"""
		data = gen_metadata(self.meta, 1, 'hello', [3,4,5], (1,2), 7.0)

	def test1c_gen_meta_kwargs(self):
		"""Test metadata generation with **kwargs"""
		data = gen_metadata(self.meta, extra=True, world='Earth', planets=7, names=['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'])

	def test2a_store_meta(self):
		"""Test metadata storing"""
		thismeta = gen_metadata(self.meta)
		self.outfiles = store_metadata(thismeta, 'TestMetaData', dir='/tmp/', aspickle=True, asjson=True)

		# Files should exist
		self.assertTrue(os.path.isfile(self.outfiles['pickle']), 'Pickle file not stored correctly')
		self.assertTrue(os.path.isfile(self.outfiles['json']), 'JSON file not stored correctly')

		# File should be larger than len(repr(thismeta))
		inlen = len(repr(thismeta))
		self.assertTrue(os.path.getsize(self.outfiles['pickle']) > inlen, 'Pickle file too small?')
		self.assertTrue(os.path.getsize(self.outfiles['json']) > inlen, 'JSON file too small?')

	def test2b_store_meta_recover(self):
		"""Test metadata storing & reloading"""
		thismeta = gen_metadata(self.meta)
		self.outfiles = store_metadata(thismeta, 'TestMetaData', dir='/tmp/', aspickle=True, asjson=True)

		# Load all formats, compare with input dict
		for format, file in self.outfiles.iteritems():
			if (file and os.path.isfile(file)):
				self.assertTrue(os.path.isfile(file))
				inmeta = load_metadata(file, format)
				self.assertEqual(thismeta, inmeta)

class TestFITSutils(unittest.TestCase):
	def setUp(self):
		self.cards = {'hello': 'world', 'name':'testprogram'}

	def test1a_mkfitshdr(self):
		"""Test mkfitshdr calls"""
		mkfitshdr({})
		mkfitshdr({}, usedefaults=False)
		mkfitshdr(self.cards)
		mkfitshdr(self.cards, usedefaults=False)

	def test1b_mkfitdhdr_long(self):
		"""Test longer calls"""
		mkfitshdr({'filename': "reproc2_slide3"})
		mkfitshdr({'filename': "reproc2_slide3-tgMOV-0.5-cosine-0.75"})
		mkfitshdr({'filename': "reproc2_slide3-tgMOV-0.5-cosine-0.75/"})
		mkfitshdr({'filename': "reproc2_slide3-tgMOV-0.5-cosine-0.75/0_141132-463_Zernike_reconstruction.fits"})

class TestParsestr(unittest.TestCase):
	def setUp(self):
		pass

	def test1a_known_simpl(self):
		"""Test on simple list"""

		rng_calc = parse_range_str("1,2,3,4")
		rng_expect = [1,2,3,4]
		self.assertEqual(rng_calc, rng_expect)

	def test1b_known(self):
		"""Test range expansion on known positive list"""

		rng_calc = parse_range_str("6 - 10,1,2,3,7-10,19-10")
		rng_expect = [6, 7, 8, 9, 10, 1, 2, 3, 7, 8, 9, 10]
		self.assertEqual(rng_calc, rng_expect)

	def test1c_negative(self):
		"""Test on negative list"""

		rng_calc = parse_range_str("-5 - -1, 0, 5, 8, 9, 10, 100-105")
		rng_expect = [-5, -4, -3, -2, -1, 0, 5, 8, 9, 10, 100, 101, 102, 103, 104, 105]
		self.assertEqual(rng_calc, rng_expect)

	def test2a_negative_cust(self):
		"""Test on negative list, custom rsep"""

		rng_calc = parse_range_str("-5 = -1, 0=5", rsep="=")
		rng_expect = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
		self.assertEqual(rng_calc, rng_expect)

		rng_calc = parse_range_str("-5 . -1, 0.5", rsep=".")
		rng_expect = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
		self.assertEqual(rng_calc, rng_expect)

	def test4a_fail_input(self):
		"""Test on illegal input"""

		self.assertRaises(ValueError, parse_range_str, "0a")
		self.assertRaises(ValueError, parse_range_str, "5-10a")
		self.assertRaises(ValueError, parse_range_str, "5aa10a")

	def test4a_fail_sep(self):
		"""Test for illegal rsep, sep"""

		self.assertRaises(ValueError, parse_range_str, "0", rsep='0')
		self.assertRaises(ValueError, parse_range_str, "0", sep='0')

class TestTokenize(unittest.TestCase):
	def setUp(self):
		pass

	def test1a_test_known(self):
		"""Test known list of tokenization"""
		# Setup data for this test
		self.strl1 = ["uni-20110900_0000.ppm.png",
		"uni-20110900_0820.ppm.png",
		"uni-20110900_0930.ppm.png"]
		self.strl1_tok = ["0000","0820","0930"]
		self.strl1_notok = ["00","82","93"]

		tp = find_uniq(self.strl1, tokenize=True, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in self.strl1]
		self.assertEqual(tokend, self.strl1_tok)

		tp = find_uniq(self.strl1, tokenize=False, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in self.strl1]
		self.assertEqual(tokend, self.strl1_notok)

	def test1b_test_synt(self):
		"""Test synthetic list of tokenization"""

		# Longer synthetic data, unique subset are dates and file indices
		self.strl2 = ["unibrain-frame-201109%02d_%04d.ppm.png" % (dd, idx % 1000) for dd in xrange(8,12) for idx in xrange(0,128*512,7*512)]
		el0 = self.strl2[0]

		tp = find_uniq(self.strl2, tokenize=True, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in self.strl2]

		self.assertEqual(el0[:tp[0]], "unibrain-frame-")
		self.assertEqual(el0[tp[1]:], ".ppm.png")

		tp = find_uniq(self.strl2, tokenize=False, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in self.strl2]

		self.assertEqual(el0[:tp[0]], "unibrain-frame-201109")
		self.assertEqual(el0[tp[1]:], ".ppm.png")

	def test1c_test_alluniq(self):
		"""Test list of strings which are all unique"""

		self.strl5 = ["%02d_%04d" % (dd, idx % 1000) for dd in xrange(8,12) for idx in xrange(0,128*512,7*512)]
		el0 = self.strl5[0]

		tp = find_uniq(self.strl5, tokenize=True, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in self.strl5]

		self.assertEqual(el0[:tp[0]], "")
		self.assertEqual(el0[tp[1]:], "")

		tp = find_uniq(self.strl5, tokenize=False, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in self.strl5]

		self.assertEqual(el0[:tp[0]], "")
		self.assertEqual(el0[tp[1]:], "")

	def test1d_test_known(self):
		"""Test two identical strings, prefix and postfix should be empty"""

		# Setup data for this test
		strl = ['unibrain-frame-201109%d_%d.ppm.png', 'unibrain-frame-201109%d_%d.ppm.png']
		el0 = strl[0]

		tp = find_uniq(strl, tokenize=True, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in strl]

		self.assertEqual(el0[:tp[0]], "")
		self.assertEqual(el0[tp[1]:], "")

		tp = find_uniq(strl, tokenize=False, tokens=['.', '-', '_'])
		tokend = [s[tp[0]:tp[1]] for s in strl]

		self.assertEqual(el0[:tp[0]], "")
		self.assertEqual(el0[tp[1]:], "")

	def test2a_test_vary_raise(self):
		"""Test if varying string length raises ValueError"""

		# Varying length data
		self.strl3 = ["unibrain-frame-201109%d_%d.ppm.png" % (dd, idx % 1000) for dd in xrange(8,12) for idx in xrange(0,128*512,7*512)]

		self.assertRaises(ValueError, find_uniq, self.strl3, tokenize=True, tokens=['.', '-', '_'])
		self.assertRaises(ValueError, find_uniq, self.strl3, tokenize=False, tokens=['.', '-', '_'])

	def test1d_test_random(self):
		"""Test random list of strings for robustness"""

		# Random strings
		import numpy as np
		self.strl4 = [''.join((chr(int(i)) for i in np.random.random((128))*255)) for i in range(50)]

		# Test two modes. Output is probably garbage
		tp = find_uniq(self.strl4, tokenize=True, tokens=['.', '-', '_'])
		tp = find_uniq(self.strl4, tokenize=False, tokens=['.', '-', '_'])


if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())
