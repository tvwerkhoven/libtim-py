#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@file ao.py Adaptive optics calibration/calculation routines
@author Tim van Werkhoven
@date 20130626
@copyright Copyright (c) 2013 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
"""

import im
import warnings
import numpy as np
import pylab as plt

def comp_influence(inflmeas, inflact, binfac=None, singval=1.0, add_offset=False, verb=0):
	"""
	Calculate the system influence matrix and its inverse from calibration 
	data, using **singval** as cut-off for the singular value decomposition.

	Given:

		(1)   error + measmat = [infl_mat, offset] . [actmat, 1]

	with **measmat** and **actmat** known, we solve for **infl_mat**:

		(2)   (error + measmat) . [actmat, 1]^+ = [infl_mat, offset]

	where **actmat^+** is the pseudo inverse of **actmat** using all 
	singular values.

	We check this solution with:

		(3a)  epsilon = measmat - infl_mat . actmat
		(3b)  epsilon = measmat^+ . (infl_mat . actmat) - identity
		(3c)  measmat_rec = infl_mat . actmat
		(3d)  actmat_rec = measmat . infl_mat^+

	where **epsilon** should be small compared to the mean of **measmat**.

	Finally we obtain the control matrix from the inverse of the influence 
	matrix, **infl_mat^+** by singular value decomposing **infl_mat** 
	using a limited set of singular values for regularisation. We check this 
	solution with:

		(4)   measmat_rec = infl_mat . actmat
		(5)   measmat . infl_mat^+ = actmat

	@param [in] inflmeas Influence measurement data matrix
	@param [in] inflact Influence actuation data matrix
	@param [in] binfac Bin measurement data (inflmeas, inflact) by this factor for speed
	@param [in] verb Verbosity
	@return Dictionary with keys inflmat, epsilon, offsetvec, ctrlmat, svdcomps (as U, s, Vh))
	"""

	infl = {}

	if (binfac):
		# Check if this binfac divides the size to an integer
		if (inflmeas.shape[1]/(binfac*1.0) == int(inflmeas.shape[1]/(binfac*1.0))):
			inflmeas = sum([inflmeas[::, i::binfac] for i in range(binfac)])/binfac
		else:
			warnings.warn("Not binning, factor not divisor!")
	
	# Add offset vector if requested
	if (add_offset):
		inflact = np.hstack([ inflact, np.ones_like(inflact0[:,0:1]) ])

	# 2. Compute the influence matrix
	inflmat = np.dot(np.linalg.pinv(inflact), inflmeas)
	
	# 3a. Check the quality as a fraction of mismatch for influence measurements
	infl['epsilon'] = np.abs(inflmeas - np.dot(inflact, inflmat)).mean() / np.abs(inflmeas).mean()

	# Remove offset mode, use for static correction
	infl['offsetvec'] = inflmat[-1:]
	# @bug @todo This was required for data read from disk?
	infl['inflmat'] = (inflmat[:-1]).newbyteorder('=')

	infl['svdcomps'] = svd_U, svd_s, svd_Vh = np.linalg.svd(infl['inflmat'], full_matrices=False)
	nmodes = lambda s, perc: np.argwhere(s.cumsum()/s.sum() >= perc)[0][0] if s.sum() else 0
	svd_s_red = svd_s.copy()
	svd_s_red[nmodes(svd_s, singval)+1:] = np.inf

	infl['ctrlmat'] = np.dot(svd_Vh.T.conj(), np.dot(np.diag(1.0/svd_s_red), svd_U.T.conj()))

	# 3c. Check the quality in reconstructing the actuation matrix
	infl['actmat_rec'] = np.dot(inflmeas - infl['offsetvec'], infl['ctrlmat'])

	infl['offcorrvec'] = offvec = -np.dot(infl['offsetvec'], infl['ctrlmat'])

	if (np.abs(offvec).max() > 1.0):
		print "Offset correction actuation, reg.: ", " ".join("%.2g" % e for e in offvec.flat)
		warnings.warn("Saturation in offset correction vector, static optical errors are large or calibration failed!")
		raw_input("Continue..")
	elif (verb):
		print "Offset correction actuation, reg.: ", " ".join("%.2g" % e for e in offvec.flat)

	return infl

def inspect_influence(inflact, inflmeas, apt_mask, infldat=None, what="all", fignum0=1000):
	"""
	Given influence data, inspect it
	"""
	
	if infldat == None: 
		infldat = comp_influence(inflmeas, inflact)
	svd_U, svd_s, svd_Vh = infldat['svdcomps']
	offsetvec = infldat['offsetvec'].ravel()

	dshape = apt_mask.shape
	apt_mask_c = im.mk_rad_mask(*dshape) < 0.9

	if (what in ["all", "singval"]):
		plt.figure(fignum0+100); plt.clf()
		plt.title("Singvals (n=%d, sum=%.4g, c=%.4g)" % (len(svd_s), svd_s.sum(), svd_s[0]/svd_s[-1]))
		plt.xlabel("Mode [#]")
		plt.ylabel("Singular value [AU]")
		plt.plot(svd_s)

		plt.figure(fignum0+101); plt.clf()
		plt.title("Singvals [log] (n=%d, sum=%.4g, c=%.4g)" % (len(svd_s), svd_s.sum(), svd_s[0]/svd_s[-1]))
		plt.xlabel("Mode [#]")
		plt.ylabel("Singular value [AU]")
		plt.semilogy(svd_s)
		raw_input("Continue...")

	if (what in ["all", "offset"]):
		# Init new empty image
		thisvec = np.zeros(dshape)
		# Fill image with masked vector data
		thisvec[apt_mask] = offsetvec.ravel()
		thisvec[apt_mask == False] = offsetvec.mean()
		imin, imax = thisvec[apt_mask_c].min(), thisvec[apt_mask_c].max()
		plt.figure(fignum0+200); plt.clf()
		plt.xlabel("X [pix]")
		plt.ylabel("Y [pix]")
		plt.title("Offset phase (system aberration)")
		plt.imshow(thisvec, vmin=imin, vmax=imax)
		plt.colorbar()
		raw_input("Continue...")

	if (what in ["all", "actrec"]):
		plt.figure(fignum0+250); plt.clf()
		plt.title("Actuation matrix reconstruction")
		plt.xlabel("Actuator [id]")
		plt.ylabel("Measurement [#]")
		plt.imshow(infldat['actmat_rec'])
		plt.colorbar()
		raw_input("Continue...")

	if (what in ["all", "inflact"]):
		# Init new empty image
		thisvec = np.zeros(dshape)
		for idx, inflvec in enumerate(infldat['inflmat']):
			thisvec[apt_mask] = inflvec.ravel()
			thisvec[apt_mask == False] = inflvec.mean()
			imin, imax = thisvec[apt_mask_c].min(), thisvec[apt_mask_c].max()

			plt.figure(fignum0+300); plt.clf()
			plt.xlabel("X [pix]")
			plt.ylabel("Y [pix]")
			plt.title("Actuator %d influence" % (idx))
			plt.imshow(thisvec, vmin=imin, vmax=imax)
			plt.colorbar()
			if (raw_input("Continue [b=break]...") == 'b'): break

	if (what in ["all", "sysmodes"]):
		# Init new empty image
		thisbase = np.zeros(dshape)
		for idx, base in enumerate(svd_Vh):
			thisbase[apt_mask] = base.ravel()
			thisbase[apt_mask == False] = base.mean()
			imin, imax = thisbase[apt_mask_c].min(), thisbase[apt_mask_c].max()

			plt.figure(fignum0+400); plt.clf()
			plt.xlabel("X [pix]")
			plt.ylabel("Y [pix]")
			plt.title("DM base %d, s=%g (%g)" % (idx, svd_s[idx], svd_s[idx]/svd_s.sum()))
			plt.imshow(thisbase, vmin=imin, vmax=imax)
			plt.colorbar()
			if (raw_input("Continue [b=break]...") == 'b'): break

	if (what in ["all", "measrec"]):
		measmat_rec = np.dot(inflact[:,:-1], infldat['inflmat'])
		thismeasrec = np.zeros(dshape)
		for idx, measvec_rec in enumerate(measmat_rec):
			thismeasrec[apt_mask] = measvec_rec.ravel()
			thismeasrec[apt_mask == False] = measvec_rec.mean()
			imin, imax = thismeasrec[apt_mask_c].min(), thismeasrec[apt_mask_c].max()

			plt.figure(fignum0+500); plt.clf()
			plt.xlabel("X [pix]")
			plt.ylabel("Y [pix]")
			plt.title("influence meas. %d rec" % (idx))
			plt.imshow(thismeasrec, vmin=imin, vmax=imax)
			plt.colorbar()
			if (raw_input("Continue [b=break]...") == 'b'): break

	raw_input("Will now exit and close windows...")
	for fnum in [100, 101, 200, 250, 300, 400, 500]:
		plt.figure(fignum0 + fnum); plt.close()
