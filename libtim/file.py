#!/usr/bin/env python
# encoding: utf-8
"""
@file file.py
@brief File I/O utilities

@package libtim.file
@brief File I/O utilities
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120403

This module provides some file IO functions.
"""

#=============================================================================
# Import libraries here
#=============================================================================

import matplotlib.image as mpimg
import numpy
import pyfits
import json
import cPickle
import string
import os

import unittest

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def read_file(fpath, dtype=None, **kwargs):
	"""
	Try to read datafile at **fpath**.

	Try to read **fpath** and return contents. If **dtype** is set, force reading routines with this datatype, otherwise guess from extension or simply try.

	Supported datatypes:
	- FITS through pyfits.getdata
	- NPY through numpy.load
	- NPZ through numpy.load
	- CSV through numpy.loadtxt(delimiter=',')
	- JSON through json.load
	- pickle through cPickle.load

	@todo Add region of interest loading? Should be dimension independent...

	@param [in] fpath Path to a file
	@param [in] dtype Datatype to read. If absent, guess.
	@param [in] **kwargs Extra parameters passed on directly to read function
	@return Data from file, usually as numpy.ndarray
	"""

	# Check datatype, if not set: detect from file extension
	if (dtype == None):
		dtype = os.path.splitext(fpath)[1].lower()[1:]

	# Check correct read function
	if (dtype == 'fits'):
		# FITS needs pyfits
		return pyfits.getdata(fpath, **kwargs)
	elif (dtype == 'npy'):
		# NPY needs numpy
		return numpy.load(fpath, **kwargs)
	elif (dtype == 'npz'):
		# NPZ needs numpy
		datadict = numpy.load(fpath, **kwargs)
		if (len(datadict.keys()) > 1):
			print >> sys.stderr, "Warning! Multiple files stored in archive '%s', returning only the first" % (fpath)
		return datadict[datadict.keys()[0]]
	elif (dtype == 'csv'):
		# CSV needs Numpy.loadtxt
		return numpy.loadtxt(fpath, delimiter=',', **kwargs)
	elif (dtype == 'pickle'):
		fp = open(fpath, 'r')
		dat = cPickle.load(fp, **kwargs)
		fp.close()
		return dat
	elif (dtype == 'json'):
		fp = open(fpath, 'r')
		dat = json.load(fp, **kwargs)
		fp.close()
		return dat
	else:
		# Anything else should work with PIL's imread(). If not, it will throw anyway so we don't need to check
		return mpimg.imread(fpath, **kwargs)

def store_file(fpath, data, **kwargs):
	"""
	Store **data** to disk at **fpath**.

	Inverse of read_file(). Datatype is guessed from fpath.

	Supported datatypes:
	- FITS through pyfits.writeto
	- NPY through numpy.save
	- NPZ through numpy.savez
	- CSV through numpy.savetxt
	- PNG through matplotlib.image.imsave
	- JSON through json.dump
	- pickle through cPickle.dump

	@param [in] data Data to store. Should be something that converts to a numpy.ndarray
	@param [in] fpath Full path to store to
	@param [in] **kwargs Extra parameters passed on directly to write function
	@returns Path the data is stored to, when successful
	"""

	# Guess dtype from filepath
	dtype = os.path.splitext(fpath)[1].lower()[1:]

	# Check correct write function
	if (dtype == 'fits'):
		# FITS needs pyfits
		pyfits.writeto(fpath, data, **kwargs)
	elif (dtype == 'npy'):
		# NPY needs numpy
		numpy.save(fpath, data, **kwargs)
	elif (dtype == 'npz'):
		# NPY needs numpy
		numpy.savez(fpath, data, **kwargs)
	elif (dtype == 'csv'):
		# CSV needs Numpy.loadtxt
		numpy.savetxt(fpath, data, delimiter=',', **kwargs)
	elif (dtype == 'png'):
		mpimg.imsave(fpath, data, **kwargs)
	elif (dtype == 'json'):
		fp = open(fpath, 'w')
		json.dump(data, fp, indent=2, **kwargs)
		fp.close()
	elif (dtype == 'pickle'):
		fp = open(fpath, 'w')
		cPickle.dump(data, fp, **kwargs)
		fp.close()
	else:
		raise ValueError("Unsupported filetype '%s'" % (dtype))

	return fpath

def read_files(flist, dtype=None):
	"""
	@deprecated Use '[read_file(f) for f in flist]' instead
	"""
	raise DeprecationWarning("Use '[read_file(f) for f in flist]' instead")

def filenamify(str):
	"""
	Convert any string into a valid filename.

	Given an input string, convert it to a reasonable filename by rejecting unknown characters. Valid characters are ASCII letters, digits and -_.().

	Internally this uses:

	\code
	>>> "-_.()%s%s" % (string.ascii_letters, string.digits))
	'-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
	\endcode

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

class TestReadWriteFiles(unittest.TestCase):
	def setUp(self):
		self.dataformats = ['fits', 'npy', 'npz', 'csv', 'png']
		self.metaformats = ['json', 'pickle']
		self.allformats = self.dataformats + self.metaformats
		self.files = []

	def tearDown(self):
		"""Delete files produces in this test"""
		for file in self.files:
# 			print "Removing temp files", self.files
			if (file and os.path.isfile(file)):
				os.remove(file)

	def test1a_filenamify(self):
		"""Test filenamify calls"""
		self.assertEqual(filenamify('hello world'), 'hello_world')

	def test1b_read_file_calls(self):
		"""Test read_file calls"""

		# These should all raise an IOerror
		with self.assertRaisesRegexp(IOError, "No such file or directory"):
			read_file('nonexistent.file', None)

		for fmt in self.allformats:
			with self.assertRaisesRegexp(IOError, "No such file or.*"):
				read_file('nonexistent.file', fmt)

	def test1c_write_file(self):
		"""Test write_file"""
		# Generate data
		sz = (67, 47)
		data1 = N.random.random(sz).astype(N.float)
		data2 = (N.random.random(sz)*255).astype(N.uint8)
		meta1 = {'meta': 'hello world', 'len': 123, 'payload': [1,4,14,4,111]}

		# Store as all formats
		for fmt in self.dataformats:
			fpath = store_file('/tmp/TestReadWriteFiles_data1.'+fmt, data1)
			self.files.append(fpath)
			fpath = store_file('/tmp/TestReadWriteFiles_data2.'+fmt, data2)
			self.files.append(fpath)

		for fmt in self.metaformats:
			fpath = store_file('/tmp/TestReadWriteFiles_meta1.'+fmt, meta1)
			self.files.append(fpath)

	def test2a_read_file_data(self):
		"""Test read_file reconstruction"""
		# Generate data
		sz = (67, 47)
		data1 = N.random.random(sz).astype(N.float)
		data2 = (N.random.random(sz)*255).astype(N.uint8)
		meta1 = {'meta': 'hello world', 'len': 123, 'payload': [1,4,14,4,111]}

		# Store as all formats
		for fmt in self.dataformats:
			fpath = store_file('/tmp/TestReadWriteFiles_data1.'+fmt, data1)
			self.files.append(fpath)
			fpath = store_file('/tmp/TestReadWriteFiles_data2.'+fmt, data2)
			self.files.append(fpath)

		# Try to read everything again
		for fmt in self.dataformats:
			read1 = read_file('/tmp/TestReadWriteFiles_data1.'+fmt)
			read2 = read_file('/tmp/TestReadWriteFiles_data2.'+fmt)
			# PNG loses scaling, ignore
			if fmt not in ['png']:
				self.assertTrue(N.allclose(data1, read1))
				self.assertTrue(N.allclose(data2, read2))

		for fmt in self.metaformats:
			fpath = store_file('/tmp/TestReadWriteFiles_meta1.'+fmt, meta1)
			self.files.append(fpath)

		for fmt in self.metaformats:
			read1 = read_file('/tmp/TestReadWriteFiles_meta1.'+fmt)
			self.assertEqual(meta1, read1)



if __name__ == "__main__":
	import numpy as N
	import sys
	sys.exit(unittest.main())
