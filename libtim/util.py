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

#============================================================================
# Import libraries here
#============================================================================

import sys
import os
import hashlib
import pyfits
from time import asctime, gmtime, time, localtime
import cPcikle as pickle
import json

#============================================================================
# Defines
#============================================================================

#============================================================================
# Routines
#============================================================================

def find_uniq(strlist, tokenize=True, tokens=['.', '-', '_', '/']):
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
	- SHA1 hex digest of current script (sha1(sys.argv[0]))
	- path & SHA1 digest of python interpreter used (sys.executable)

	and additionally save everything in *args and **kwargs.

	This is intended to store all program execution parameters to disk such that this batch can later be reproduced, and the origin of the output can be traced back.

	@param [in] metadata Dict of other values to store
	@param [in] *args Additional values to store
	@param [in] **kwargs Additional key-value pairs to store
	@returns Dictionary containing all values
	@see store_metadata, load_metadata
	"""

	# Hash a file without reading it fully
	# <http://stackoverflow.com/a/4213255>
	sha1_h = hashlib.sha1()
	with open(sys.argv[0],'rb') as f:
		for chunk in iter(lambda: f.read(128*sha1_h.block_size), ''):
			 sha1_h.update(chunk)
	fhash = sha1_h.hexdigest()

	sha1_h = hashlib.sha1()
	with open(sys.executable,'rb') as f:
		for chunk in iter(lambda: f.read(128*sha1_h.block_size), ''):
			 sha1_h.update(chunk)
	ihash = sha1_h.hexdigest()

	# Start metadata dictionary with pre-set values
	metadict = {'curdir': os.path.realpath(os.path.curdir),
		'program': sys.argv[0],
		'argv': " ".join(sys.argv[1:]),
		'epoch': time(),
		'utctime': asctime(gmtime(time())),
		'localtime':asctime(localtime(time())),
		'hostid': os.uname()[1],
		'progsize': os.stat(sys.argv[0]).st_size,
		'progmtime': os.path.getmtime(sys.argv[0]),
		'progctime': os.path.getctime(sys.argv[0]),
		'sha1script': fhash,
		'interppath': sys.executable,
		'sha1interp': ihash}

	grev = git_rev(sys.argv[0])
	if (grev):
		metadict.update({'revision': grev})

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
	@returns Dict of files written to in format:path syntax
	@see gen_metadata, load_metadata
	"""
	# Prepend directory to output path
	basepath = os.path.join(dir, basename)

	# Store output files here
	outfiles = {}

	if (aspickle):
		pickle_file = basepath + "_meta.pickle"
		fp = open(pickle_file, 'w')
		cPickle.dump(metadict, fp)
		fp.close()
		outfiles['pickle'] = pickle_file
	if (asjson):
		json_file = basepath + "_meta.json"
		fp = open(json_file, 'w')
		json.dump(metadict, fp, indent=2)
		fp.close()
		outfiles['json'] = json_file

	return outfiles

def load_metadata(infile, format='json'):
	"""
	Load metadata from **infile**.

	Load metadata stored in general by store_metadata(), specify format withis **format**.

	@param [in] infile filepath to read
	@param [in] format Format of filepath (json or pickle)
	@return Dict of metadata, like the input of store_metadata()
	@see gen_metadata, store_metadata
	"""

	metad = {}

	if (format.lower() == 'json'):
		fp = open(infile, 'r')
		metad = json.load(fp)
		fp.close()
	elif (format.lower() == 'pickle'):
		fp = open(infile, 'r')
		metad = cPickle.load(fp)
		fp.close()

	return metad


def mkfitshdr(cards=None, usedefaults=True):
	"""
	Make a FITS file header of all arguments supplied in the dict **cards**.

	If **usedefaults** is set, also add default header items:
	- Program filename and pasth (from sys.argv[0])
	- Current working dir
	- Program filesize, mtime and ctime
	- Git revision of executable (if available)
	- epoch (time())
	- utctime / localtime
	- hostid

	@params [in] cards Dict containing key=value pairs for the header
	@params [in] usedefaults Also store default parameters in header
	@return pyfits header object
	"""

	clist = pyfits.CardList()

	if (usedefaults):
		clist.append(pyfits.Card(key='progname', 
								value=os.path.basename(sys.argv[0]),
								comment='Program filename') )
		clist.append(pyfits.Card(key='progpath', 
								value=os.path.dirname(sys.argv[0]),
								comment='Program path') )
		grev = git_rev(sys.argv[0])
		if (grev):
			clist.append(pyfits.Card(key='gitrev', 
								value=grev,
								comment='Program git revision') )
		clist.append(pyfits.Card(key='progsize', 
								value=os.path.getsize(sys.argv[0]),
								comment='Program filesize (bytes)') )
		clist.append(pyfits.Card(key='mtime', 
								value=os.path.getmtime(sys.argv[0]),
								comment='Program last last modification time') )
		clist.append(pyfits.Card(key='ctime', 
								value=os.path.getctime(sys.argv[0]),
								comment='Program metadata change time' ) )
		clist.append(pyfits.Card(key='curdir', 
								value=os.path.realpath(os.path.curdir),
								comment='Current working dir') )
		clist.append(pyfits.Card(key='epoch', value=time(),
								comment='Current seconds since epoch from time.time()') )
		# No comments for the last two fields because they are too large
		clist.append(pyfits.Card(key='utctime', value=asctime(gmtime(time()))) )
		clist.append(pyfits.Card(key='loctime', value=asctime(localtime(time()))) )
		clist.append(pyfits.Card(key='hostid', value=os.uname()[1],
								comment='Hostname from os.uname()') )

	if (cards):
		for key, val in cards.iteritems():
			clist.append(pyfits.Card(key, val) )

	return pyfits.Header(cards=clist)

def svn_rev(fpath):
	"""
	Query and return svn revision of a certain path.

	@param [in] fpath Path to investigate. Can be filename, in which case only the path will be use
	@returns Output of `svn info`
	"""

	try:
		fdir = os.path.dirname(fpath)
	except:
		raise ValueError("Cannot get directory for '%s'" % fpath)

	import subprocess
	cmd = ['svn', 'info']
	proc = subprocess.Popen(cmd, cwd=fdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out = proc.communicate()[0]
	# Take only stdout

	if (out):
		for line in out.splitlines():
			if "Revision" in line:
				rev = line[10:]
				return rev

	return ""

def git_rev(fpath):
	"""
	Query and return git revision of a certain path.

	@param [in] fpath Path to investigate. Can be filename, in which case only the path will be use
	@returns Output of `git describe --always --dirty`
	"""

	try:
		fdir = os.path.dirname(fpath)
	except:
		raise ValueError("Cannot get directory for '%s'" % fpath)

	# Put in try clause in case git is not available
	try:
		import subprocess
		cmd = ['git', 'describe', '--always', '--dirty']
		proc = subprocess.Popen(cmd, cwd=fdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		# Take only stdout
		out = proc.communicate()[0]
		rev = out.rstrip()
	except:
		rev = ""

	return rev

def parse_uptime(upstr, version='OSX'):
	"""
	Calculate single uptime scalar from uptime(1) output.

	Input examples:
	(from http://caterva.org/projects/parse_uptime_with_sed/)
	# NetBSD 1.6.2
	9:55AM up 1 min, 1 user, load averages: 0.11, 0.12, 0.14
	10:55AM up 1 hr, 1 user, load averages: 0.11, 0.12, 0.14
	12:00PM up 6 days, 14:42, 1 user, load averages: 0.25, 0.17, 0.10
	10:26AM up 32 mins, 1 user, load averages: 0.11, 0.12, 0.14
	10:56AM up 1:03, 1 user, load averages: 0.25, 0.24, 0.18
	10:56AM up 6 days, 32 mins, 1 user, load averages: 0.25, 0.24, 0.18
	8:53AM up 23 hrs, 1 user, load averages: 0.13, 0.13, 0.09
	9:54AM up 1 day, 1 user, load averages: 0.34, 0.29, 0.16
	9:55AM up 1 day, 1 min, 1 user, load averages: 0.44, 0.31, 0.17
	10:55AM up 1 day, 1 hr, 1 user, load averages: 0.44, 0.31, 0.17

	# Debian 5.0
	20:45:11 up 0 min,  1 user,  load average: 0.55, 0.17, 0.06
	20:45:17 up 1 min,  1 user,  load average: 0.51, 0.17, 0.06
	20:51:39 up 7 min,  1 user,  load average: 0.50, 0.16, 0.06
	20:55:56 up 11 min,  1 user,  load average: 0.02, 0.08, 0.05
	22:23:27 up  1:39,  1 user,  load average: 0.00, 0.00, 0.00

	# MacOSX 10.6.2
	21:52  up 6 days,  4:49, 4 users, load averages: 0.26 0.32 0.38
	
	# Mac OSX 10.7
	18:30 up 1 min, 2 users, load averages: 0.71 0.20 0.07
	14:00 up 36 mins, 2 users, load averages: 0.82 0.51 0.33
	12:45 up 23:21, 4 users, load averages: 4.51 5.34 5.12
	13:30 up 1 day, 6 mins, 4 users, load averages: 0.71 0.60 1.02
	1:57 up 2 days, 12:19, 5 users, load averages: 0.56 0.51 0.61
	14:30 up 1 day, 1:05, 2 users, load averages: 0.55 0.48 0.55

	Input syntax:
	[N day[s], ][HH:MM|MM min[s]]

	@param [in] upstr uptime(1) output
	@param [in] version uptime version (OSX, Linux, ...)
	@returns Tuple of (localtime as string, uptime in days, nuser, tuple of load averages)
	"""

	# String cannot be shorter than this
	if (len(upstr) < len("0:0 up 1:00, 1 user, load averages: 0 0 0")):
		raise ValueError("Input '%s' too short!" % (upstr))

	# First extract easy stuff, time is the first space-separated string, 
	# load averages are the last 3 space-separated strings.
	upwords = upstr.split()
	loadavgs = tuple(float(l.strip(',')) for l in upwords[-3:])
	localtime = upwords[0]
	nuser = int(upwords[-7])

	# Find user

	# Select substring from ' up ' to the comma before ' user'
	upsub = upstr.split(' up ')[1].split(' user')[0]
	upsub = upsub[:upsub.rfind(",")]

	# Check if there are days
	days = 0
	if ('day' in upsub):
		days, upsub = upsub.split('day')
		# Skip comma and possible s
		upsub = upsub[2:]

	# Check for hours
	if ('hr' in upsub):
		hours, minutes = upsub.split("hr")[0], 0
	# Check if there are minutes
	elif ('min' in upsub):
		hours, minutes = 0, upsub.split("min")[0]
	elif (':' in upsub):
		hours, minutes = upsub.split(":")
	else:
		hours, minutes = 0, 0
	
	# Calculate time
	uptime =  float(days) + float(hours)/24. + float(minutes)/1440

	return (localtime, uptime, nuser, loadavgs)
