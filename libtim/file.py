#!/usr/bin/env python
# encoding: utf-8
"""
This module provides some file IO functions
"""

##  @file file.py
# @author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
# @date 20120403
#
# Created by Tim van Werkhoven on 2012-04-03.
# Copyright (c) 2012 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
#
# This file is licensed under the Creative Commons Attribution-Share Alike
# license versions 3.0 or higher, see
# http://creativecommons.org/licenses/by-sa/3.0/

## @package file
# @brief Library for file I/O
# @author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
# @date 20120403
#
# This module provides some file IO functions.

#=============================================================================
# Import libraries here
#=============================================================================

import matplotlib.image as mpimg

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
	"""Read files from <flist>. If <dtype> is set, force reading routines
	with this datatype, otherwise guess from extension or simply try.

	This routine will always return the data in a list, even if <flist>
	only contains one filename.

	@param [in] flist List of file paths to read
	@param [in] dtype Datatype to read. If absent, guess.
	@return List of of data matrices.
	"""

	# If we have only one file, put in single-element list
	if (flist.__class__ == str):
		flist = [flist]

	# Check datatype, if not set: detect from file extension
	if (dtype == None):
		dtype = os.path.splitext(flist[0])[1].lower()[1:]

	# Check correct read function
	if (dtype == 'fits'):
		# FITS needs pyfits
		read_func = pyfits.getdata
	else:
		# Anything else should work with PIL's imread(). If not, it will throw anyway so we don't need to check
		read_func = mpimg.imread

	# Read files
	return [read_func(path) for path in flist]

if __name__ == "__main__":
	import sys
	import unittest
	sys.exit(unittest.main())
