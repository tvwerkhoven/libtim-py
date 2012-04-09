#!/usr/bin/env python
# encoding: utf-8
"""
@file util.py
@brief Miscellaneous string manipulation functions

@package libtim.util
@brief Miscellaneous string manipulation functions
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120403

This module provides some miscellaneous utility functions for parsing strings,
filenames, making headers etc.
"""

#=============================================================================
# Import libraries here
#=============================================================================

import sys
import os
import hashlib
import pyfits
from time import asctime, gmtime, time, localtime
import unittest

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def find_uniq(strlist, tokenize=True, tokens=['.', '-', '_']):
	"""
	Find shortest substring that uniquely identifies all strlist entries.

	In a list of strings **strlist** of equal length (e.g. filenames), find the shortest continuous part of the string that uniquely identifies each item in the list. If **tokenize** is True, the string is split only at any of the characters in **tokens**, otherwise it is split at any letter.

	Example, given these strings:

		unibrain-frame-20110916_0000.ppm.png
		unibrain-frame-20110916_0001.ppm.png
		unibrain-frame-20110916_0002.ppm.png

	the shortest unique id would be

		0
		1
		2

	in the filenames (the rest are similar). If **tokenize** is True, this will be

		0000
		0001
		0002

	@param [in] strlist List of strings to find unique subset for
	@param [in] tokenize Split by tokens instead of characters
	@param [in] tokens List of tokens to use for **tokenize**
	@return Two indices which denote the start and end of the unique substring as a tuple.
	@see find_tok_pos() used to tokenize input strings
	"""

	# If we only have one item, the whole string is unique
	if (len(strlist) < 2):
		return (0, len(strlist[0]))

	# If <tokenize>, then split the string in tokens, separated by <tokens>.
	# Otherwise, split by character.
	if (tokenize):
		### Find length of prefix in first and last string

		# Find the positions of all tokens in the first file, use full string
		# as prefix as initial guess, this will be trimmed later
		tokpos_f = find_tok_pos(strlist[0], tokens)
		pre_idx = len(tokpos_f)-1

		# Find length of postfix in first and last string. Get right side of
		# tokens (add one). Use full string as postfix, trim later
		tokpos_b = find_tok_pos(strlist[0], tokens, past=True)
		post_idx = 0

		### Loop over all files. For each consecutive pair of files, check if
		# the prefix and postfix substrings are equal. If not, trim the prefix
		# and postfix length by one token and continue until they are equal.
		for idx in xrange(len(strlist)-1):
			if (len(strlist[idx]) != len(strlist[idx+1])):
				raise ValueError("Input string list are not of equal length!")
			while (strlist[idx][:tokpos_f[pre_idx]] != strlist[idx+1][:tokpos_f[pre_idx]]):
				pre_idx -= 1
			while (strlist[idx][tokpos_b[post_idx]:] != strlist[idx+1][tokpos_b[post_idx]:]):
				post_idx += 1

		# If pre_idx and post_idx are still the same, all strings are the
		# same, in which case the above strategy of trimming prefix and
		# postfix sequentially fails. Fix this by manually setting boundaries
		if (pre_idx == len(tokpos_f)-1 and post_idx == 0):
			return (0, len(strlist[0]))

		prelen = tokpos_f[pre_idx]
		# Subtract one from postlen position because we exclude the token
		# itself
		postlen = tokpos_b[post_idx]-1
		return (prelen, postlen)
	else:
		# Find unique prefix and postfix between element 0 and -1
		# Guess prefix length as full string
		prelen = len(strlist[0])
		# Guess initial postfix length as full string
		postlen = 0

		# At this point, prelen and postlen can only get shorter:
		for idx in xrange(1, len(strlist)-1):
			if (len(strlist[idx]) != len(strlist[idx+1])):
				raise ValueError("Input string list are not of equal length!")
			while (strlist[idx][:prelen] != strlist[idx+1][:prelen]):
				prelen -= 1
			while (strlist[idx][postlen:] != strlist[idx+1][postlen:]):
				postlen += 1

		# If prelen and postlen are still the same, all strings are the
		# same, in which case the above strategy of trimming prefix and
		# postfix sequentially fails. Fix this by manually setting boundaries
		if (prelen == len(strlist[0]) and postlen == 0):
			return (0, len(strlist[0]))

		return (prelen, postlen)

def find_tok_pos(tokstr, tokens=['.', '-', '_'], rev=False, past=True):
	"""
	Find positions of **tokens** in **tokstr**.

	Given a string **tokstr**, return a sorted list of the positions of all the tokens in **tokens**. If **rev**(erse) is True, search from the back instead of the front. If **past** is True, store the position of the token plus one so we exclude it in substrings.

	@param [in] tokstr String to tokenize
	@param [in] tokens List of tokens to find
	@param [in] rev Reverse search order
	@param [in] past Give position+1 instead of position (i.e. 'past' the token)
	"""
	# Reverse string to search from back
	if (rev):
		tokstr = tokstr[::-1]

	# Init list with 0 as a boundary
	tokpos = [0]
	for t in tokens:
		# If token does not exist, skip
		if (tokstr.find(t) == -1):
			continue
		# Start
		cpos = -1
		for t in tokstr.split(t)[:-1]:
			cpos += len(t)+1
			if (past):
				tokpos.append(cpos+1)
			else:
				tokpos.append(cpos)


	# Append length+1 as marker
	if (past):
		tokpos.append(len(tokstr)+1)
	else:
		tokpos.append(len(tokstr))

	# Sort for extra kudos
	tokpos.sort()
	return tokpos

def parse_range_str(rstr, sep=",", rsep="-", offs=0):
	"""
	Expand numerical ranges in **rstr** to all integers.

	Given a string **rstr** representing a range of integers, expand it to all integers in that range. For example, the string

		instr = '1,2,3,7-10,19-25'

	would expand to

		[1, 2, 3, 7, 8, 9, 19, 20, 21, 22, 23, 24, 25]

	i.e.:

	\code
	>>> parse_range_str("1,2,3,7-10,19-25")
	[1, 2, 3, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25]
	\endcode

	@param [in] rstr String to expand
	@param [in] sep Separator to use
	@param [in] rsep Range indicator to use
	@param [in] offs Offset to add to output
	@returns List of integers in expanded range
	"""
	if (rsep == sep):
		raise ValueError("<sep> and <rsep> cannot be identical")

	# int(rsep) and int(sep) should raise, otherwise, something is wrong
	rflag = 0
	try: a = int(rsep); rflag = 1
	except: pass
	try: a = int(sep); rflag = 1
	except: pass
	if (rflag): raise ValueError("<rsep> and <sep> should not parse to int")

	els = []
	# Split input string around <sep>
	for el in rstr.split(sep):
		# If <rsep> is in this <el> (like '7-10 '), this is a range that needs expansion. In that case, split the the element around <rsep>, and calculate range(el[0], el[1]+1)
		# Note that <resp> should not be the first character (i.e. (-5-0)) to
		# accomodate for negative start range
		el = el.strip()
		if (rsep in el[1:]):
			spl_idx = el[1:].find(rsep)+1
			els.extend( range(int(el[:spl_idx]), int(el[spl_idx+1:])+1) )
		else:
			els.append( int(el) )
	# Apply offset and return
	return [i+offs for i in els]

def gen_metadata(metadata, *args, **kwargs):
	"""
	Generate metadata dict to use for identifying program executions.

	Generate metadata dictionary with data about current program execution. **metadata** should be a dict holding extra information, furthermore these default values will be added as well:
	- current filename (**sys.argv[0]**)
	- program arguments (**sys.argv[1:]**)
	- time / date (as epoch, utc, localtime)
	- size of current executable
	- SHA1 hex digest of current executable (sha1(sys.argv[0]))

	and additionally save everything in *args and **kwargs.

	This is intended to store all program execution parameters to disk such that this batch can later be reproduced, and the origin of the output can be traced back.

	@param [in] metadata Dict of other values to store
	@param [in] *args Additional values to store
	@param [in] **kwargs Additional key-value pairs to store
	@returns Dictionary containing all values
	@see store_metadata
	"""

	# Hash a file without reading it fully
	# <http://stackoverflow.com/a/4213255>
	sha1_h = hashlib.sha1()
	with open(sys.argv[0],'rb') as f:
		for chunk in iter(lambda: f.read(128*sha1_h.block_size), ''):
			 sha1_h.update(chunk)
	fhash = sha1_h.hexdigest()

	# Start metadata dictionary with pre-set values
	metadict = {'program': sys.argv[0],
		'argv': " ".join(sys.argv[1:]),
		'epoch': time(),
		'utctime': asctime(gmtime(time())),
		'localtime':asctime(localtime(time())),
		'hostid': os.uname()[1],
		'progsize': os.stat(sys.argv[0]).st_size,
		'sha1digest': fhash}

	# Add user-supplied values
	metadict.update(metadata)

	# Add *args and **kwargs
	metadict["args"] = {}
	for (i, arg) in enumerate(args):
		metadict["args"][i] = arg
	metadict["kwargs"] = {}
	for key in kwargs:
		metadict["kwargs"][str(key)] = kwargs[key]

	return metadict

def store_metadata(metadict, basename, dir='./', aspickle=False, asjson=True):
	"""
	Store metadata in **metadict** to disk.

	Given a dictionary, store it to disk in various formats. Currently pickle and JSON are supported, although the latter is preferred.

	This function is intended to be used in conjunction with gen_metadata() to store data about a data processing job.

	@param [in] metadict Dictionary of values to store.
	@param [in] basename Basename to store data to.
	@param [in] dir Output directory
	@param [in] aspickle Store as pickle format
	@param [in] asjson Store as JSON format
	@see gen_metadata
	"""
	# Prepend directory to output path
	basepath = os.path.join(dir, basename)

	if (aspickle):
		import cPickle
		fp = open(basepath + "_meta.pickle", 'w')
		cPickle.dump(metadict, fp)
		fp.close()
	if (asjson):
		import json
		fp = open(basepath + "_meta.json", 'w')
		json.dump(metadict, fp, indent=2)
		fp.close()

def mkfitshdr(cards):
	"""
	Make a FITS file header of all arguments supplied in the dict **cards**.

	Besides **cards**, also add default header items:
	- Program name (sys.argv[0])
	- epoch (time()
	- utctime / localtime
	- hostname

	@params [in] cards Dict containing key=value pairs for the header
	@return pyfits header object
	"""
	# Init list
	clist = pyfits.CardList()

	# Add default fields
	clist.append(pyfits.Card('prog', sys.argv[0]) )
	clist.append(pyfits.Card('epoch', time()) )
	clist.append(pyfits.Card('utctime', asctime(gmtime(time()))) )
	clist.append(pyfits.Card('loctime', asctime(localtime(time()))) )

	# Add custom fields
	for key, val in cards.iteritems():
		clist.append(pyfits.Card(key, val) )

	return pyfits.Header(cards=clist)

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

		with self.assertRaisesRegexp(ValueError, "invalid literal"):
			rng_calc = parse_range_str("0a")
		with self.assertRaisesRegexp(ValueError, "invalid literal"):
			rng_calc = parse_range_str("5-10a")
		with self.assertRaisesRegexp(ValueError, "invalid literal"):
			rng_calc = parse_range_str("5aa10a")

	def test4a_fail_sep(self):
		"""Test for illegal rsep, sep"""

		with self.assertRaisesRegexp(ValueError, "should not parse to int"):
			rng_calc = parse_range_str("0", rsep='0')
		with self.assertRaisesRegexp(ValueError, "should not parse to int"):
			rng_calc = parse_range_str("0", sep='0')


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

		with self.assertRaisesRegexp(ValueError, ".*not of equal length.*"):
			find_uniq(self.strl3, tokenize=True, tokens=['.', '-', '_'])

		with self.assertRaisesRegexp(ValueError, ".*not of equal length.*"):
			find_uniq(self.strl3, tokenize=False, tokens=['.', '-', '_'])

	def test1d_test_random(self):
		"""Test random list of strings for robustness"""

		# Random strings
		self.strl4 = [''.join((chr(int(i)) for i in N.random.random((128))*255)) for i in range(50)]

		# Test two modes. Output is probably garbage
		tp = find_uniq(self.strl4, tokenize=True, tokens=['.', '-', '_'])
		tp = find_uniq(self.strl4, tokenize=False, tokens=['.', '-', '_'])


if __name__ == "__main__":
	import numpy as N
	import sys
	sys.exit(unittest.main())
