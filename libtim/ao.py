#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@file ao.py Adaptive optics calibration/calculation routines
@author Tim van Werkhoven
@date 20130626
@copyright Copyright (c) 2013 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
"""

import im
import zern
import warnings
import numpy as np
import pylab as plt
from os.path import join as pjoin

def comp_influence(measmat, actmat, binfac=None, singval=1.0, nmodes=0, add_offset=False, use_zernike=False, verb=0):
	"""
	Calculate the system influence matrix and its inverse from calibration 
	data, using **singval** as cut-off for the singular value decomposition.

	Given:

		(1)   measmat = [infl_mat, offset] . [actmat, 1]

	with **measmat** and **actmat** known, we solve for **infl_mat**:

		(2)   measmat . [actmat, 1]^+ = [infl_mat, offset]

	where **actmat^+** is the pseudo inverse of **actmat** using all 
	singular values.

	Finally we obtain the control matrix from the inverse of the influence 
	matrix, **infl_mat^+** by singular value decomposing **infl_mat** 
	using a limited set of singular values for regularisation:

		(4a)  ctrl_mat = infl_mat^+
		(4b)  U, s, V^H = svd(infl_mat)
		(4c)  ctrl_mat = V . diag(1/s) . U^T

	where ctrl_mat is (n_act, n_meas)

	We check this solution with:

		(3a)  epsilon = measmat - infl_mat . actmat
		(3b)  measmat_rec = infl_mat . actmat
		(3c)  actmat_rec = measmat . infl_mat^+

	where **epsilon** should be small compared to the mean of **measmat**.

	@param [in] measmat Calibration measurements, size (n_data, n_meas)
	@param [in] actmat Calibration actuations (n_act, n_meas)
	@param [in] binfac Bin measurement data (measmat) by this factor
	@param [in] singval Amount of singular value to use (relative, [0, 1])
	@param [in] nmodes Number of modes to use in inversion (cut-off if negative)
	@param [in] verb Verbosity
	@return Dictionary with keys inflmat, epsilon, offsetmeas, offsetact, ctrlmat, svdcomps (as U, s, Vh))
	"""

	infl = {}

	# Check matrix size, measmat should be (n_data, n_meas), actmat should 
	# be (n_act, n_meas). We should have:
	# n_data > n_meas
	# n_meas > n_act
	n_data, n_meas1 = measmat.shape
	n_act, n_meas2 = actmat.shape
	assert (n_data > n_meas1), "Measurement matrix wrong, transposed?"
	assert (n_meas2 > n_act), "Actuation matrix wrong, transposed?"
	assert (n_meas2 == n_meas1), "Matrices incompatible, not the same number of measurements"

	if (binfac):
		# Check if this binfac divides the size to an integer
		if (measmat.shape[0]/(binfac*1.0) == int(measmat.shape[0]/(binfac*1.0))):
			measmat = sum([measmat[i::binfac] for i in range(binfac)])/binfac
		else:
			warnings.warn("Not binning, factor not a divisor!")
	
	# Add offset vector if requested
	if (add_offset):
		actmat = np.vstack([ actmat, np.ones_like(actmat[0:1]) ])

	# 2. Compute the influence matrix
	inflmat = np.dot(measmat, np.linalg.pinv(actmat))
	
	# 3a. Check the quality as a fraction of mismatch for influence measurements
	infl['infleps'] = np.abs(measmat - np.dot(inflmat, actmat)).mean() / np.abs(measmat).mean()

	# Remove offset mode, use for static correction
	infl['offsetmeas'] = inflmat[:, -1:]
	infl['inflmat'] = inflmat[:, :-1]

	# 4b. Decompose influence matrix with SVD
	infl['svdcomps'] = svd_U, svd_s, svd_Vh = np.linalg.svd(infl['inflmat'], full_matrices=False)
	# Compute how many modes are required to get *at least* perc singular value power (perc in [0, 1])
	nmodes_f = lambda s, perc: np.argwhere(s.cumsum()/s.sum() >= min(perc, 1.0))[0][0]+1 if s.sum() else 0
	if (nmodes < 0):
		nmodes = len(svd_s) + nmodes
	elif (nmodes > 0):
		pass
	else:
		nmodes = nmodes_f(svd_s, singval)

	svd_s_red = svd_s.copy()
	svd_s_red[nmodes:] = np.inf
	infl['usedmodes'] = nmodes
	infl['totmodes'] = len(svd_s)

	# 4c. Compute (regularized) control matrix
	# V . diag(1/s) . U^H
	# We take the conjugate transpose here in case the input data are complex
	# See http://mathworld.wolfram.com/ConjugateTranspose.html
	infl['ctrlmat'] = np.dot(svd_Vh.conj().T, np.dot(np.diag(1.0/svd_s_red), svd_U.conj().T))

	# 3c. Check the quality in reconstructing the actuation matrix
	infl['actmat_rec'] = np.dot(infl['ctrlmat'],measmat - infl['offsetmeas'])
	infl['acteps'] = np.abs(infl['actmat_rec'] - actmat[:-1]).mean()
	infl['offsetact'] = offvec = -np.dot(infl['ctrlmat'], infl['offsetmeas'])

	if (np.abs(offvec).max() > 1.0):
		print "Offset correction actuation, reg.: ", " ".join("%.2g" % e for e in offvec.flat)
		warnings.warn("Saturation in offset correction vector, static optical errors are large or calibration failed!")
		raw_input("Continue..")
	elif (verb):
		print "Offset correction actuation, reg.: ", " ".join("%.2g" % e for e in offvec.flat)

	return infl

def comp_zernike_ctrl(inflmat, apt_mask, zerncache=None):
	"""
	Given the influence matrix of a system as phase, compute a matrix that 
	maps Zernike vectors to system control vectors.
	"""

	# This does not work for SHWFS influence matrices, which usually have 
	# <500 measurements (i.e. n_subp<250 or n_pix<500).
	assert inflmat.shape[0] > 500, "Influence matrix has less than 500 data points?"

	# Compute Zernike control matrix for this system
	if (zerncache == None):
		zerncache = zern.calc_zern_basis(20, apt_mask.shape[0]/2, modestart=2, calc_covmat=False)

	zernactmat = np.dot(np.linalg.pinv(inflmat), zerncache['modesmat'][:,apt_mask.ravel()].T)

	return zernactmat

def inspect_influence(actmat, measmat, apt_mask, infldat=None, what="all", fignum0=1000, store=False, interactive=True, outdir='./'):
	"""
	Given influence data, inspect it
	"""
	
	# Check matrix size, measmat should be (n_data, n_meas), actmat should 
	# be (n_act, n_meas). We should have:
	# n_data > n_meas
	# n_meas > n_act
	n_data, n_meas1 = measmat.shape
	n_act, n_meas2 = actmat.shape
	assert (n_data > n_meas1), "Measurement matrix wrong, transposed?"
	assert (n_meas2 > n_act), "Actuation matrix wrong, transposed?"
	assert (n_meas2 == n_meas1), "Matrices incompatible, not the same number of measurements"

	if infldat == None: 
		infldat = comp_influence(measmat, actmat)
	
	svd_U, svd_s, svd_Vh = infldat['svdcomps']
	offsetmeas = infldat['offsetmeas'].ravel()
	dshape = apt_mask.shape
	apt_mask_c = im.mk_rad_mask(*dshape) < 0.9
	plot_mask = np.ones(apt_mask.shape)
	plot_mask[apt_mask==False] = np.nan

	if (what in ["all", "singval"]):
		plt.figure(fignum0+100); plt.clf()
		plt.title("Singvals (n=%d, sum=%.4g, c=%.4g)" % (len(svd_s), svd_s.sum(), svd_s[0]/svd_s[-1]))
		plt.xlabel("Mode [#]")
		plt.ylabel("Singular value [AU]")
		plt.plot(svd_s)
		if (store): plt.savefig(pjoin(outdir, "cal_singvals.pdf"))

		plt.figure(fignum0+101); plt.clf()
		plt.title("Singvals [log] (n=%d, sum=%.4g, c=%.4g)" % (len(svd_s), svd_s.sum(), svd_s[0]/svd_s[-1]))
		plt.xlabel("Mode [#]")
		plt.ylabel("Singular value [AU]")
		plt.semilogy(svd_s)
		if (store): plt.savefig(pjoin(outdir, "cal_singvals_logy.pdf"))

		plt.figure(fignum0+102); plt.clf()
		plt.title("1-singvals cumsum (n=%d, sum=%.4g, c=%.4g)" % (len(svd_s), svd_s.sum(), svd_s[0]/svd_s[-1]))
		plt.xlabel("Mode [#]")
		plt.ylabel("Residual singval [AU]")
		plt.semilogy(1-svd_s.cumsum()/svd_s.sum())
		if (store): plt.savefig(pjoin(outdir, "cal_singvals_cumsum_logy.pdf"))
		if (interactive): raw_input("Continue...")

	if (what in ["all", "offset"]):
		# Init new empty image
		thisvec = np.zeros(dshape)
		# Fill image with masked vector data
		thisvec[apt_mask] = offsetmeas
		imin, imax = offsetmeas.min(), offsetmeas.max()
		plt.figure(fignum0+200); plt.clf()
		plt.xlabel("X [pix]")
		plt.ylabel("Y [pix]")
		plt.title("Offset phase (system aberration)")
		plt.imshow(thisvec*plot_mask, vmin=imin, vmax=imax)
		plt.colorbar()
		if (store): plt.savefig(pjoin(outdir, "cal_offset_phase.pdf"))
		if (interactive): raw_input("Continue...")

	if (what in ["all", "actrec"]):
		plt.figure(fignum0+250); plt.clf()
		plt.title("Actuation matrix reconstruction")
		vmin, vmax = actmat.min(), actmat.max()
		plt.subplot(131)
		plt.xlabel("Actuator [id]")
		plt.ylabel("Measurement [#]")
		plt.imshow(actmat.T, vmin=vmin, vmax=vmax)
		plt.subplot(132)
		plt.xlabel("Actuator [id]")
		plt.imshow(infldat['actmat_rec'].T, vmin=vmin, vmax=vmax)
		plt.colorbar()
		plt.subplot(133)
		plt.xlabel("Actuator [id]")
		plt.imshow((actmat[:-1]-infldat['actmat_rec']).T)
		plt.colorbar()
		if (store): plt.savefig(pjoin(outdir, "cal_actmat_rec.pdf"))
		if (interactive): raw_input("Continue...")

	if (what in ["all", "actmat"]):
		# Init new empty image
		thisvec = np.zeros(dshape)
		for idx, inflvec in enumerate(infldat['inflmat'].T):
			thisvec[apt_mask] = inflvec.ravel()
			imin, imax = inflvec.min(), inflvec.max()

			plt.figure(fignum0+300); plt.clf()
			plt.xlabel("X [pix]")
			plt.ylabel("Y [pix]")
			plt.title("Actuator %d influence" % (idx))
			plt.imshow(thisvec*plot_mask, vmin=imin, vmax=imax)
			plt.colorbar()
			if (store): plt.savefig(pjoin(outdir, "cal_infl_vec_%d.pdf" % (idx)))
			if (interactive): 
				if (raw_input("Continue [b=break]...") == 'b'): break

	if (what in ["all", "sysmodes"]):
		# Init new empty image
		thisbase = np.zeros(dshape)
		for idx, base in enumerate(svd_U.T):
			thisbase[apt_mask] = base.ravel()
			imin, imax = base.min(), base.max()

			plt.figure(fignum0+400); plt.clf()
			plt.xlabel("X [pix]")
			plt.ylabel("Y [pix]")
			plt.title("DM base %d, s=%g (%g)" % (idx, svd_s[idx], svd_s[idx]/svd_s.sum()))
			plt.imshow(thisbase*plot_mask, vmin=imin, vmax=imax)
			plt.colorbar()
			if (store): plt.savefig(pjoin(outdir, "cal_dm_base_vec_%d.pdf" % (idx)))
			if (interactive):
				if (raw_input("Continue [b=break]...") == 'b'): break

	if (what in ["measrec"]):
		measmat_rec = np.dot(infldat['inflmat'], actmat[:-1])
		thismeasrec = np.zeros(dshape)
		for idx, measvec_rec in enumerate(measmat_rec.T):
			thismeasrec[apt_mask] = measvec_rec.ravel()
			imin, imax = measvec_rec.min(), measvec_rec.max()

			plt.figure(fignum0+500); plt.clf()
			plt.xlabel("X [pix]")
			plt.ylabel("Y [pix]")
			plt.title("influence meas. %d rec" % (idx))
			plt.imshow(thismeasrec*plot_mask, vmin=imin, vmax=imax)
			plt.colorbar()
			if (store): plt.savefig(pjoin(outdir, "cal_infl_vec_%d_rec.pdf" % (idx)))
			if (interactive):
				if (raw_input("Continue [b=break]...") == 'b'): break

	raw_input("Will now exit and close windows...")
	for fnum in [100, 101, 200, 250, 300, 400, 500]:
		plt.figure(fignum0 + fnum); plt.close()
