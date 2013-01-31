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
import numpy as np
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
		data1 = np.random.random(sz).astype(np.float)
		data2 = (np.random.random(sz)*255).astype(np.uint8)
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
		data1 = np.random.random(sz).astype(np.float)
		data2 = (np.random.random(sz)*255).astype(np.uint8)
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
				self.assertTrue(np.allclose(data1, read1))
				self.assertTrue(np.allclose(data2, read2))

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
			data1 = np.random.random(sz).astype(np.float)
			data2 = (np.random.random(sz)*255).astype(np.uint8)
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
					self.assertTrue(np.allclose(data1, read1))
					self.assertTrue(np.allclose(data2, read2))

				# Try to read with ROI
				read1 = read_file(fn1, roi=thisroi)
				read2 = read_file(fn2, roi=thisroi)
				
				# Check dimensions, should be different
				self.assertNotEqual(read1.shape, data1.shape)
				self.assertNotEqual(read2.shape, data2.shape)
				self.assertEqual(read1.shape, thisshp)
				self.assertEqual(read2.shape, thisshp)

				# Check ROI data ok
				if (len(thisroi) == 2):
					r0 = slice(*thisroi[0:2])
					self.assertTrue(np.allclose(read1, data1[r0]))
					self.assertTrue(np.allclose(read2, data2[r0]))
				elif (len(thisroi) == 4):
					r0 = slice(*thisroi[0:2])
					r1 = slice(*thisroi[2:4])
					self.assertTrue(np.allclose(read1, data1[r0, r1]))
					self.assertTrue(np.allclose(read2, data2[r0, r1]))
				elif (len(thisroi) == 6):
					r0 = slice(*thisroi[0:2])
					r1 = slice(*thisroi[2:4])
					r2 = slice(*thisroi[4:6])
					self.assertTrue(np.allclose(read1, data1[r0, r1, r2]))
					self.assertTrue(np.allclose(read2, data2[r0, r1, r2]))

	def test2c_test_read_roi_bin(self):
		"""Test read_file reconstruction"""
		# Generate data
		szl = [(67,), (67, 47), (67, 47, 32)]
		roil = [(50, 60), (50, 60, 20, 40), (50, 60, 20, 40, 5, 13)]
		shapel = [(10,), (10, 20), (10, 20, 8)]

		for sz, thisroi, thisshp in zip(szl, roil, shapel):
			data1 = np.random.random(sz).astype(np.float)
			data2 = (np.random.random(sz)*255).astype(np.uint8)
			data2f = data2.astype(np.float)
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

				# Try to read with ROI
				read1 = read_file(fn1, roi=thisroi, bin=2)
				read2 = read_file(fn2, roi=thisroi, bin=2)

				
				# Check ROI/bin data is ok
				if (len(thisroi) == 2):
					# Manual RoI and binning
					r0 = slice(*thisroi[0:2])
					data1b = data1[r0][::2] + data1[r0][1::2]
					data2b = data2f[r0][::2] + data2f[r0][1::2]

					self.assertTrue(np.allclose(read1, data1b))
					self.assertTrue(np.allclose(read2, data2b))
				elif (len(thisroi) == 4):
					# Manual RoI and binning
					r0 = slice(*thisroi[0:2])
					r1 = slice(*thisroi[2:4])
					data1b = data1[r0, r1][::2, ::2] + data1[r0, r1][1::2, ::2] + data1[r0, r1][::2, 1::2] + data1[r0, r1][1::2, 1::2]
					data2b = data2f[r0, r1][::2, ::2] + data2f[r0, r1][1::2, ::2] + data2f[r0, r1][::2, 1::2] + data2f[r0, r1][1::2, 1::2]

					self.assertTrue(np.allclose(read1, data1b))
					self.assertTrue(np.allclose(read2, data2b))
				elif (len(thisroi) == 6):
					r0 = slice(*thisroi[0:2])
					r1 = slice(*thisroi[2:4])
					r2 = slice(*thisroi[4:6])
					data1b = data1[r0, r1, r2][::2, ::2, ::2] + \
						data1[r0, r1, r2][1::2,  ::2,  ::2] + \
						data1[r0, r1, r2][ ::2, 1::2,  ::2] + \
						data1[r0, r1, r2][1::2, 1::2,  ::2] + \
						data1[r0, r1, r2][ ::2,  ::2, 1::2] + \
						data1[r0, r1, r2][1::2,  ::2, 1::2] + \
						data1[r0, r1, r2][ ::2, 1::2, 1::2] +\
						data1[r0, r1, r2][1::2, 1::2, 1::2]
					data2b = data2f[r0, r1, r2][::2, ::2, ::2] + \
						data2f[r0, r1, r2][1::2,  ::2,  ::2] + \
						data2f[r0, r1, r2][ ::2, 1::2,  ::2] + \
						data2f[r0, r1, r2][1::2, 1::2,  ::2] + \
						data2f[r0, r1, r2][ ::2,  ::2, 1::2] + \
						data2f[r0, r1, r2][1::2,  ::2, 1::2] + \
						data2f[r0, r1, r2][ ::2, 1::2, 1::2] +\
						data2f[r0, r1, r2][1::2, 1::2, 1::2]	

					self.assertTrue(np.allclose(read1, data1b))
					self.assertTrue(np.allclose(read2, data2b))

if __name__ == "__main__":
	import sys
	sys.exit(unittest.main())
