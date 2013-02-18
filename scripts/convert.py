#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file convert.py -- convert one data format to another
@author Tim van Werkhoven
@date 20120913
@copyright Copyright (c) 2012 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)

Convert files between dataformats. Inspired by Imagemagick's convert.

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

# Import my own utilities
import libtim as tim
from libtim.file import read_file, store_file
import numpy as np
import sys, os
import argparse

# Define some contants
AUTHOR = "Tim van Werkhoven <werkhoven@strw.leidenuniv.nl>"
DATE = "20120913"

def main():
	## Parse arguments check options
	(parser, args) = parsopts()

	# Print some debug output
	if (args.verb > 1):
		print args
	
	# Load all data. If only one file, flatten
	roit = tuple(args.roi)
	if (len(args.infiles) > 1):
		indata = np.r_[ [ read_file(f, roi=roit, squeeze=args.squeeze) for f in args.infiles] ]
	else: 
		indata = read_file(args.infiles[0], roi=roit, squeeze=args.squeeze)

	# Store to disk, add options depending on file type
	if ('png' in os.path.splitext(args.outfile)[1].lower()):
		if (args.verb > 1): print "Output as PNG, using cmap==%s" % (args.cmap)
		store_file(args.outfile, indata, cmap=args.cmap)
	else:
		if (args.verb > 1): print "Other output"
		store_file(args.outfile, indata)

def parsopts():
	"""Parse program options, check sanity and return results"""
	import argparse
	parser = argparse.ArgumentParser(description='Convert datafiles between formats. Uses libtim.file.read_file() and store_file().', epilog='Comments & bugreports to %s' % (AUTHOR))

	parser.add_argument('infiles', metavar='IN', nargs="+",
						help='input file(s)')
	parser.add_argument('--outfile', metavar='OUT', required=True,
						help='output file')
	
	g0 = parser.add_argument_group("""Input options when loading data. ROI can be 2, 4 or 6 arguments for 1, 2 or 3-dimensional data. ROI should be specified as [low, high] for each dimension.""")

	g0.add_argument('--roi', nargs='*', type=int,
						help='region of interest for input files.')
	g0.add_argument('--squeeze', action='store_true', default=True, 
						help='trim 1-element axes from input files (np.squeeze).')

	g1 = parser.add_argument_group("""PNG output options. Often used colormaps: RdYlBu, YlOrBr, gray""")
	g1.add_argument('--cmap', type=str, default="YlOrBr", 
						help="color map to use [YlOrBr]")

	parser.add_argument('-v', dest='debug', action='append_const', const=1,
						help='increase verbosity')
	parser.add_argument('-q', dest='debug', action='append_const', const=-1,
						help='decrease verbosity')

	args = parser.parse_args()

	# Check & fix some options
	checkopts(parser, args)

	# Return results
	return (parser, args)

def checkopts(parser, args):
	"""Check program options sanity"""
	# Check verbosity
	args.verb = 0
	if (args.debug):
		args.verb = sum(args.debug)
	if (len(args.roi) not in [2,4,6]):
		print "Error: roi should be 2, 4 or 6 arguments."
		parser.print_usage()
		exit(-1)

# This must be the final part of the file, code after this won't be executed
if __name__ == "__main__":
	sys.exit(main())

### EOF
