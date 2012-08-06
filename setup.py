#!/usr/bin/env python
# encoding: utf-8
"""
setup.py -- setup file for the libtim module

Created by Tim van Werkhoven (werkhoven@strw.leidenuniv.nl) on 2010-05-19
Copyright (c) 2010--2012 Tim van Werkhoven. All rights reserved.
"""

from distutils.core import setup, Command
from unittest import TextTestRunner, TestLoader

cmdclasses = dict()

class TestCommand(Command):
    """Runs the unittests for libtim"""
    
    description = "Runs the unittests for libtim"

    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        loader = TestLoader()
        t = TextTestRunner()
        # Run all test*py files in libtim subdirectory
        t.run(loader.discover('libtim'))

# 'test' is the parameter as it gets added to setup.py
cmdclasses['test'] = TestCommand

# Setup
setup(cmdclass = cmdclasses,
    name = 'libtim',
	version = 'v0.1.2',
	description = 'Miscellaneous image manipulation tools library',
	keywords = 'wavefronts, zernike, FFT, cross-correlation',
	author = 'Tim van Werkhoven',
	author_email = 'werkhoven@strw.leidenuniv.nl',
	url = 'http://work.vanwerkhoven.org/',
	license = "GPL",
	packages = ['libtim'])
