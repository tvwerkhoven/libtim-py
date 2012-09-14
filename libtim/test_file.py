#!/usr/bin/env python
# encoding: utf-8
"""
@file test_file.py
@brief Test libtim.file

@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120806

Unittests for libtim.file
"""

from file import *
import numpy as N
import unittest

class TestReadWriteFiles(unittest.TestCase):
	def setUp(self):
		self.dataformats = ['fits', 'npy', 'npz', 'csv', 'png']
		self.metaformats = ['json', 'pickle']
		self.allformats = self.dataformats + self.metaformats
		self.files = []
		for file in self.files:
# 			print "Removing temp files", self.files
			if (file and os.path.isfile(file)):
				os.remove(file)

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

		# Do the same for metaformats
		for fmt in self.metaformats:
			fpath = store_file('/tmp/TestReadWriteFiles_meta1.'+fmt, meta1)
			self.files.append(fpath)

		for fmt in self.metaformats:
			read1 = read_file('/tmp/TestReadWriteFiles_meta1.'+fmt)
			self.assertEqual(meta1, read1)

	def test2b_test_read_roi(self):
		"""Test read_file reconstruction"""
		# Generate data
		szl = [(67,), (67, 47), (67, 47, 32)]
		roil = [(50, 60), (50, 60, 20, 40), (50, 60, 20, 40, 5, 12)]
		shapel = [(10,), (10, 20), (10, 20, 7)]

		for sz, thisroi, thisshp in zip(szl, roil, shapel):
			data1 = N.random.random(sz).astype(N.float)
			data2 = (N.random.random(sz)*255).astype(N.uint8)
			# Store as all formats
			for fmt in self.dataformats:
				if fmt in ['png']:
					continue
				if fmt in ['csv'] and len(sz) > 2:
					continue
				fn1 = '/tmp/TestReadWriteFiles_data1_'+str(sz)+'.'+fmt
				fn2 = '/tmp/TestReadWriteFiles_data2_'+str(sz)+'.'+fmt

				fpath = store_file(fn1, data1)
				self.files.append(fpath)
				fpath = store_file(fn2, data2)
				self.files.append(fpath)

				# Try to read everything again
				read1 = read_file(fn1)
				read2 = read_file(fn2)
				# PNG loses scaling, ignore
				if fmt not in ['png']:
					self.assertTrue(N.allclose(data1, read1))
					self.assertTrue(N.allclose(data2, read2))

				# Try to read with ROI
				read1 = read_file(fn1, roi=thisroi)
				read2 = read_file(fn2, roi=thisroi)
				# Check dimensions, should be different
				self.assertNotEqual(read1.shape, data1.shape)
				self.assertNotEqual(read2.shape, data2.shape)
				self.assertEqual(read1.shape, thisshp)
				self.assertEqual(read2.shape, thisshp)



if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())