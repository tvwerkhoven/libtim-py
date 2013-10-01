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
import numpy as np
import pyfits
import json
import cPickle
import string
import os, shutil
import fnmatch
import time

#=============================================================================
# Defines
#=============================================================================

#=============================================================================
# Routines
#=============================================================================

def read_file(fpath, dtype=None, roi=None, squeeze=False, bin=None, **kwargs):
	"""
	Try to read datafile at **fpath**.

	Try to read **fpath** and return contents. If **dtype** is set, force 
	reading routines with this data type, otherwise guess from extension or 
	simply try.

	Supported datatypes:
	- FITS through pyfits.getdata
	- NPY through numpy.load
	- NPZ through numpy.load
	- CSV through numpy.loadtxt(delimiter=',')
	- pickle through cPickle.load
	- JSON through json.load

	All other formats are read with matplotlib.image.imread(), which links 
	to PIL for anything except PNG.

	Additionally, a region of interest can be select. This is currently 
	supported for 1, 2 and 3-dimensional data. The format should be (low, 
	high) for each dimension in sequence. If `high' == -1, use all data. 
	For each dimension, the RoI will be translated to a slice as:

		roisl0 = slice(roi[0], None if (roi[1] == -1) else roi[1])

	After selecting a region of interest, the matrix can be squeeze()'d to 
	get rid of unit-element axes. This can be useful when loading PNG files 
	with useless color information.

	Pixel binning is available through the **bin** parameter. This single
	integer scalar indicates how  many neighbouring pixels will be summed
	together. N.B. When binning,  data will always be returned as np.float 
	due to potential overflow  situations when using the native file 
	datatype.

	Raises RuntimeError if RoI, bin and/or data dimensions don't match up.

	@todo Make region of interest dimension independent

	@param [in] fpath Path to a file
	@param [in] dtype Datatype to read. If absent, guess.
	@param [in] roi Region of interest to read from file, (0low, 0high, 1low, 1high, ..nlow, nhigh) or None
	@param [in] squeeze squeeze() array after selecting roi
	@param [in] bin Bin scalar for all dimensions, should be integer and multiple of shape.
	@param [in] **kwargs Extra parameters passed on directly to read function

	@return RoI'd and binned data from file, usually as numpy.ndarray
	"""

	# Check datatype, if not set: detect from file extension
	if (dtype == None):
		dtype = os.path.splitext(fpath)[1].lower()[1:]

	# Check correct read function
	if (dtype == 'fits'):
		# FITS needs pyfits
		data = pyfits.getdata(fpath, **kwargs)
	elif (dtype == 'npy'):
		# NPY needs numpy
		data = np.load(fpath, **kwargs)
	elif (dtype == 'npz'):
		# NPZ needs numpy
		datadict = np.load(fpath, **kwargs)
		if (len(datadict.keys()) > 1):
			print >> sys.stderr, "Warning! Multiple files stored in archive '%s', returning only the first" % (fpath)
		data = datadict[datadict.keys()[0]]
	elif (dtype == 'csv'):
		# CSV needs Numpy.loadtxt
		data = np.loadtxt(fpath, delimiter=',', **kwargs)
	elif (dtype == 'pickle'):
		fp = open(fpath, 'r')
		data = cPickle.load(fp, **kwargs)
		fp.close()
		# Return immediately, no ROI applicable
		return data
	elif (dtype == 'ppm' or dtype == 'pgm' or dtype == 'pbm'):
		data = read_ppm(fpath, **kwargs)
	elif (dtype == 'json'):
		fp = open(fpath, 'r')
		data = json.load(fp, **kwargs)
		fp.close()
		# Return immediately, no ROI applicable
		return data
	else:
		# Anything else should work with PIL's imread(). If not, it will throw anyway so we don't need to check
		data = mpimg.imread(fpath, **kwargs)

	data = roi_data(data, roi, squeeze)
	data = bin_data(data, bin)

	return data

def roi_data(data, roi=None, squeeze=False):
	"""
	Select a region of interest **roi** from **data**. If **squeeze** is 
	set, get rid of unit-element axes.

	@param [in] data Data to roi, can be 1, 2 or 3-dimensional.
	@param [in] roi Region of interest to read from file, (0low, 0high, 1low, 1high, ..nlow, nhigh) or None
	@param [in] squeeze squeeze() array after selecting roi
	@return Slice of data
	"""

	if (roi != None):
		if (len(roi) != data.ndim*2):
			raise RuntimeError("ROI (n=%d) does not match with data dimension (%d)!" % (len(roi), data.ndim))
		elif (len(roi) == 2):
			roisl0 = slice(roi[0], None if (roi[1] == -1) else roi[1])
			data = data[roisl0]
		elif (len(roi) == 4):
			roisl0 = slice(roi[0], None if (roi[1] == -1) else roi[1])
			roisl1 = slice(roi[2], None if (roi[3] == -1) else roi[3])
			data = data[roisl0, roisl1]
		elif (len(roi) == 6):
			roisl0 = slice(roi[0], None if (roi[1] == -1) else roi[1])
			roisl1 = slice(roi[2], None if (roi[3] == -1) else roi[3])
			roisl2 = slice(roi[4], None if (roi[5] == -1) else roi[5])
			data = data[roisl0, roisl1, roisl2]
		else:
			raise RuntimeError("This many dimensions is not supported by ROI!")
		if (squeeze):
			data = data.squeeze()
	
	return data

def bin_data(data, binfac=None):
	"""
	Bin **data** by a integer factor **binfac** in all dimensions.

	@param [in] data Data to bin, should be 1, 2 or 3-dimensional
	@param [in] binfac Factor to bin by, should be integer and >0
	@return Binned data
	"""

	binfac = int(binfac)

	# If data.shape is not divisable by binfac, skip
	if (data.shape/np.r_[binfac] != data.shape/np.r_[1.0*binfac]).any():
		return

	# If binfac is legal, start binning
	if (binfac != None and int(binfac) == binfac and binfac > 0):
		ibin = int(binfac)
		data = data.astype(np.float)
		if data.ndim == 1:
			data = np.sum(data[i::binfac] for i in range(binfac))
		elif data.ndim == 2:
			data = sum(data[i::binfac, j::binfac] for i in range(binfac) for j in range(binfac))
		elif data.ndim == 3:
			data = sum(data[i::binfac, j::binfac, k::binfac] for i in range(binfac) for j in range(binfac) for k in range(binfac))
		else:
			raise RuntimeError("This many dimensions is not supported by binning")
	return data


def read_ppm(fpath, endian='big'):
	"""
	Read binary or ASCII PGM/PPM/PBM file and return data. 16bit binary data can be interpreted as big or little endian, see <https://en.wikipedia.org/wiki/Netpbm_format#16-bit_extensions>

	@param [in] fpath File path
	@param [in] endian Endianness of the data. Binary PGM is usually big endian.
	"""
	
	fp = open(fpath, 'r')
	
	# Read magic number. P4, P5, P6 for binary, P1, P2, P3 for ASCII
	magic = fp.readline().strip()
	
	if (magic not in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6')):
		raise RuntimeError("Magic number wrong!")
	
	if (magic not in ('P2', 'P5')):
		raise NotImplementedError("Only Netpbm grayscale files (PGM) are supported")
	
	# Read size, possibly after comments
	size = fp.readline()
	while (size[0] == "#"):
		size = fp.readline()
	
	sizes = size.strip().split()
	size0, size1 = int(sizes[0]), int(sizes[1])
	
	# Read maximum value in file
	maxval = fp.readline()
	while (maxval[0] == "#"):
		maxval = fp.readline()
	
	maxval = float(maxval)
	bpp = int(np.ceil(np.log2(maxval)/8.0)*8)
	if (bpp not in (8, 16)):
		raise NotImplementedError("Only 8 and 16-bit files are supported (this file: %d)" % (bpp))

	if (magic in ('P1', 'P2', 'P3')):
		# Read all data as text
		imgdata = fp.read()
		imgarr = np.fromstring(imgdata, dtype=int, sep=' ')
	else:
		# Read data as string, convert to numpy array
		imgdata = fp.read(size0*size1*bpp/8)

		if (bpp == 8):
			imgarr = np.fromstring(imgdata, dtype=np.uint8).astype(np.uint16)
		elif (bpp == 16):
			# Convert string to ints byte by byte, but then convert those 
			# numbers to uint16 for processing afterwards
			imgarr0 = np.fromstring(imgdata[::2], dtype=np.uint8).astype(np.uint16)
			imgarr1 = np.fromstring(imgdata[1::2], dtype=np.uint8).astype(np.uint16)
			if (endian == 'big'):
				imgarr = 256*imgarr0 + imgarr1
			else:
				imgarr = imgarr0 + 256*imgarr1
	
	# Shape in proper dimension, and change origin to match 
	# matplotlib.image.imread
	imgarr.shape = (size1, size0)
	
	return imgarr[::-1]
	
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
		np.save(fpath, data, **kwargs)
	elif (dtype == 'npz'):
		# NPY needs numpy
		np.savez(fpath, data, **kwargs)
	elif (dtype == 'csv'):
		# CSV needs Numpy.loadtxt
		np.savetxt(fpath, data, delimiter=',', **kwargs)
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

def backup_file(path):
	"""
	Given a path (which can be anything that can be moved), append .bakX with X the lowest numeric suffix that does not exist, then move the path to that name.

	@param [in] path Path to move
	@return Moved path
	"""

	newpath = path + '.bak000000'
	for i in xrange(1000000):
		newpath = path + '.bak%06d' % (i)
		if (not os.path.exists(newpath)):
			break

	os.rename(path, newpath)
	return newpath


def read_files(flist, dtype=None):
	"""
	@deprecated Use '[read_file(f) for f in flist]' instead
	"""
	raise DeprecationWarning("Use '[read_file(f) for f in flist]' instead")

def read_from_dir(ddir, n=-1, purge=True, glob="*", dry=False, movedir=False):
	"""
	Read files from a directory, then remove them.

	We always wait for n+1 frames, and ignore the last one. This is to ensure that we don't copy/read/process frames that are being written. (This works if the files are properly timestamped.).

	@param [in] ddir Directory to read files from
	@param [in] n Number of files to read (-1 for all)
	@param [in] purge Delete all files in **ddir** after reading (also in dry)
	@param [in] glob Pattern the files will be filtered against
	@param [in] dry Don't read data, only return filenames
	@param [in] movedir Before reading (or returning a list of files), move the files to this directory (if set). Create if necessary.
	@return List of files
	"""

	# print "Calling read_from_dir(%s, n=%d, purge=%d, glob=%s, dry=%d, movedir=%s" % (ddir, n, purge, glob, dry, movedir)

	# List all files
	flist = os.listdir(ddir)

	# Select only files that match 'glob'
	filtlist = fnmatch.filter(flist, glob)

	# Wait until we have enough files if we asked a specific amount
	cycle = 0
	sleeptime = 0.1
	while (n != -1 and len(filtlist) < n+1):
		cycle += 1
		if (cycle % 10 == 0):
			n_got = len(filtlist)
			rate = n_got / (cycle * sleeptime)
			eta = float("inf")
			if (rate): eta = (n+1-n_got) / rate
			print "read_from_dir(): still waiting for files, got %d/%d, eta: %g sec" % (n_got, n+1, eta)
			#print "read_from_dir(): got: ", filtlist

		flist = os.listdir(ddir)
		filtlist = fnmatch.filter(flist, glob)
		time.sleep(sleeptime)

	# Always take one extra frame, and ignore this one. This is to ensure that we don't copy/read/process frames that are being written. (The list is ordered alphabetically, which works if the files are properly timestamped)
	filtlist = sorted(filtlist)[-n-1:-1]

	# If move is set, move files to that directory before returning the files (or filenames)
	if (movedir):
		# Create directory if it does not exist
		if (not os.path.isdir(movedir)):
			os.makedirs(movedir)
		# Now move all files to this directory
		for f in filtlist:
# 			print "moving %s (%d) to %s (%d)" % (os.path.join(ddir, f), os.path.exists(os.path.join(ddir, f)), movedir, os.path.isdir(movedir))
# 			print "shutil.copy2(os.path.join(%s, %s)=%s, %s)" % (ddir, f, os.path.join(ddir, f), movedir)
			shutil.move(os.path.join(ddir, f), movedir)
		# Update ddir, because all files are now in movedir
		ddir = movedir

	pathlist = [os.path.join(ddir,f) for f in filtlist]

	# Mak list mask if n != -1, but don't alter pathlist, we need it to purge
	fmask = slice(None)
	if (n != -1): fmask = slice(-n, None)

	# Read files (if not dry), return results
	if (dry):
		retl = pathlist[fmask]
	else:
		retl = [read_file(f) for f in pathlist[fmask]]

	# Purge if requested
	if (purge):
		for f in pathlist:
			os.remove(f)

	return retl

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

