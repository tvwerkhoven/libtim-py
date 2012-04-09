#!/usr/bin/env python
# encoding: utf-8
"""
setup.py -- setup file for the libtim module

Created by Tim van Werkhoven (werkhoven@strw.leidenuniv.nl) on 2010-05-19
Copyright (c) 2010--2012 Tim van Werkhoven. All rights reserved.
"""
import sys

# Try importing to see if we have NumPy available (we need this)
try:
	import numpy
	from numpy.distutils.core import setup, Extension
	from numpy.distutils.misc_util import Configuration
except:
	print "Could not load NumPy (numpy.distutils.{core,misc_util}), required by this package. Aborting."
	sys.exit(1)

# Setup
setup(name = 'libtim',
	version = 'v0.1.2',
	description = 'Miscellaneous image manipulation tools library',
	keywords = 'wavefronts, zernike, FFT, cross-correlation',
	author = 'Tim van Werkhoven',
	author_email = 'werkhoven@strw.leidenuniv.nl',
	url = 'http://work.vanwerkhoven.org/',
	license = "GPL",
	packages = ['libtim'])
