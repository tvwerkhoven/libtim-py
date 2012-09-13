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
import libtim.file
import sys
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
	
	indata = tim.file.read_file(args.infile)
	tim.file.store_file(args.outfile, indata, cmap=args.cmap)

def parsopts():
	"""Parse program options, check sanity and return results"""
	import argparse
	parser = argparse.ArgumentParser(description='Convert datafiles between formats. Uses libtim.file.read_file() and store_file().', epilog='Comments & bugreports to %s' % (AUTHOR))

	parser.add_argument('infile', metavar='IN', type=str,
						help='input file')
	parser.add_argument('outfile', metavar='OUT', type=str,
						help='output file')

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

# This must be the final part of the file, code after this won't be executed
if __name__ == "__main__":
	sys.exit(main())

### EOF
