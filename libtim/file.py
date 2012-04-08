#!/usr/bin/env python
# encoding: utf-8
"""
@package libtim.file
@brief File I/O utilities
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Copyright (c) 2012 Tim van Werkhoven
@date 20120403

This module provides some file IO functions.

This file is licensed under the Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
"""

#=============================================================================
# Import libraries here
#=============================================================================

import matplotlib.image as mpimg
import string

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def read_file(fpath, dtype=None):
	"""Read file at <fpath>. If <dtype> is set, force reading routines
	with this datatype, otherwise guess from extension or simply try.

	@param [in] fpath Path to a file
	@param [in] dtype Datatype to read. If absent, guess.
	@return Data from file
	"""

	# Check datatype, if not set: detect from file extension
	if (dtype == None):
		dtype = os.path.splitext(fpath)[1].lower()[1:]

	# Check correct read function
	if (dtype == 'fits'):
		# FITS needs pyfits
		read_func = pyfits.getdata
	else:
		# Anything else should work with PIL's imread(). If not, it will throw anyway so we don't need to check
		read_func = mpimg.imread

	# Read files
	return read_func(fpath)

def read_files(flist, dtype=None):
	raise DeprecationWarning("Use '[read_file(f) for f in flist]' instead")

def filenamify(str):
	"""
	Convert any string into a valid filename by rejecting unknown characters. Valid characters are ascii letters, digits and -_.(). (internally using %s%s" % (string.ascii_letters, string.digits)).

	@param str [in] String to convert
	@return Filtered filename
	"""
	# From <http://stackoverflow.com/a/295146>
	valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
	valid_chars = frozenset(valid_chars)

	# Replace space by _
	fbase = str.replace(' ','_')
	# Rebuild string filtering out unknown chars
	fbase = ''.join(c for c in fbase if c in valid_chars)
	return fbase


if __name__ == "__main__":
	import sys
	import unittest
	sys.exit(unittest.main())
