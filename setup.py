#!/usr/bin/env python2.5
# encoding: utf-8
"""
setup.py -- setup file for the libwfs module

Created by Tim van Werkhoven (timvanwerkhoven@gmail.com) on 2010-05-19
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
	version = 'v0.1.0',
	description = 'Personal lib, mostly wavefront-related.',
	author = 'Tim van Werkhoven',
	author_email = 'timvanwerkhoven@gmail.com',
	url = 'http://work.vanwerkhoven.org/',
	license = "GPL",
	packages = ['libtim'])
