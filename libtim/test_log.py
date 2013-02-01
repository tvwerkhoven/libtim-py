#!/usr/bin/env python
# encoding: utf-8
"""
@file log.py
@brief Terminal and file logging functionality

@package libtim.log
@brief Terminal and file logging functionality
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20090330

Logging functions to log data using prefixes, loglevels and permanent logfiles. This is probably only useful in more elaborate scripts.
"""

#==========================================================================
# Import libraries here
#==========================================================================

from log import *
import os
import unittest

#==========================================================================
# Unit tests
#==========================================================================

class TestLogger(unittest.TestCase):
	def setUp(self):
		self.msg = "test message 1234567890"
		self.longmsg = "test message 1234567890 "*10
		self.logf = "/tmp/log.py_testing.log"

	def test1a_msg(self):
		"""Test log function"""
		for v in range(7):
			try:
				log_msg(7-v, self.msg)
				log_msg(7-v, self.longmsg)
			except SystemExit:
				pass

	def test1b_msg_exit(self):
		"""Test exit raise"""
		with self.assertRaises(SystemExit):
			log_msg(ERR, self.msg)

	def test2a_logfile(self):
		"""Test logfile output"""
		init_logfile(self.logf)
		for v in range(5):
			try:
				log_msg(7-v, self.msg)
			except SystemExit:
				fd = open(self.logf, "r")
				buff = fd.read()
				fd.close()
				if (self.logf):
					os.remove(self.logf)
				self.assertGreater(len(buff), 5*len(self.msg))

if __name__ == "__main__":
	import sys, os
	sys.exit(unittest.main())
