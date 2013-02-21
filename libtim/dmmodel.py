#!/usr/bin/env python
# encoding: utf-8
"""
@file dmmodel.py
@brief Compute membrane mirror shape by solving the Poisson PDE

@package libtim.dmmodel
@brief Compute membrane mirror shape by solving the Poisson PDE
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20120903

Calculates the shape of an electrostatic bulk membrane mirror for
a given set of electrodes and voltages by solving the Poisson PDE. 

Based on response2.c which had the following copyright notice:

//
// Based on the file response.c that had the folloing copyright notice:
/*      (C) Gleb Vdovin 1997                                    */
/*      Send bug reports to gleb@okotech.com                    */
/*                                                              */
/*      Modified and cleaned in 1998                            */
/*      by Prof. Oskar von der Luehe                            */
/*      ovdluhe@kis.uni-freiburg.de                             */
// Last modified by ckeller@noao.edu on December 19, 2002
// ---------------------------------------------------------------------------


This work is licensed under the Creative Commons Attribution-Share Alike 3.0
Unported License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to Creative
Commons, 171 Second Street, Suite 300, San Francisco, California, 94105, 
USA.
"""

#===========================================================================
# Import libraries here
#===========================================================================

import numpy as np
import scipy as sp
import scipy.weave
import libtim as tim
import libtim.file
import libtim.im
# Drop to ipython during execution (call 'shell()')
from IPython import embed as shell

#===========================================================================
# Routines
#===========================================================================

def read_actmap(actmapf, verb=0):
	"""
	Read actuator map from path **actmapf**. Ensure that the returned map 
	contains integer values in the range [0,255]
	
	@param [in] actmapf Path to actuator map file
	@return Actuator map
	"""
	
	amap = libtim.file.read_file(actmapf)
	
	if (amap.ndim < 2):
		raise ValueError("Error: actuator map should be 2D!")
	elif (amap.ndim == 3):
		print "Warning: shape ", amap.ndim ,"-dimensional:", amap.shape
		# Select two dimensions that are equal and larger than 10
		if (amap.shape[0] > 10 and amap.shape[0] == amap.shape[1]):
			amap = amap[:,:,0]
		elif (amap.shape[1] > 10 and amap.shape[1] == amap.shape[2]):
			amap = amap[0,:,:]
		print "Now taking: shape ", amap.ndim ,"-dimensional:", amap.shape
	elif (amap.ndim > 3):
		raise ValueError("Error: actuator map should be 2D!")

	# If the smallest non-zero value is smaller than 1, we probably need to 
	# convert from [0,1] to [0,255]
	min_nonzero = amap[amap > 0].min()
	if (min_nonzero < 1):
		amap = np.round(amap/min_nonzero)
		if (verb > 1): print "dmmodel.read_actmap(): minimum < 1, scaling"
	
	if (amap.max() >= 255.0):
		amap[amap == 255] = 0
		if (verb > 1): print "dmmodel.read_actmap(): found 255, zeroing"
	
	return amap

def parse_volts(mirror_actmap, mirror_volts, voltfunc=lambda involt: ((involt/255.0)**2.0) / 75.7856, verb=0):
	"""
	Parse actuator map and set actuators to correct actuation. The first 
	actuator in **mirror_actmap** should have value 1.
	
	Regarding voltfunc from the original response2.c:
	// 75.7856*2 gets 3 um deflection when all voltages 
    // are set to 180; however, since the reflected wavefront sees twice
    // the surface deformation, we simply put a factor of two here

	
	@param [in] mirror_actmap DM actuator map where actuator N has pixel value N
	@param [in] mirror_volts Voltage vector for all actuators
	@param [in] voltfunc Function that maps input signal to actuation amplitude.
	"""

	for actid, actvolt in enumerate(mirror_volts):
		thismask = mirror_actmap == actid+1
		thisvolt = voltfunc(actvolt)
		mirror_actmap[thismask] = thisvolt
		if (verb > 0 and actvolt != 0):
			print "dmmodel.parse_volts(): act %d (n=%d) @ %g" % (actid, thismask.sum(), thisvolt)
	
	return mirror_actmap

def sim_dm(mirror_actmap, mirror_apt, docrop=True, verb=0):
	"""
	Simulate DM response given an actuator map and an aperture.

	The actuator map **mirror_actmap** should have value N in all elements 
	in actuator N, i.e. actuator N is at locations 
	np.argwhere(mirror_actmap == N). The first actuator should have value 1. 
	
	**mirror_apt** must have the same dimensions as mirror_actmap and the DM 
	response will be calculated at positions where mirror_apt > 0.

	If **docrop** is True, the output will be cropped to a rectangular 
	subset where **mirror_apt** is non-zero.

	@param [in] mirror_actmap Deformable mirror actuator map
	@param [in] mirror_apt Aperture map for the deformable mirror
	@param [in] docrop Crop output where **mirror_apt** > 0
	"""
	
	mirror_resp = np.zeros(mirror_actmap.shape)

	# Calculate simulation parameters
	rho = (np.cos(np.pi/mirror_resp.shape[0]) +
		 np.cos(np.pi/mirror_resp.shape[1]))/2.0;
	omega = float(2.0/(1.0 + np.sqrt(1.0 - rho*rho)))
	niter = int(2.0 * (mirror_resp.shape[0] * mirror_resp.shape[1])**0.5)
	sdif = float(0.0)
	
	# iteratively solve Poisson PDE -> cannot be vectorized, execute in C
	__COMPILE_OPTS = "-Wall -O2 -ffast-math -msse -msse2"
	
	poisson_solver = """
	#line 149 "response2.py"
	int ii=0, i=0, j=0;
	double update=0;
	double sum=0;
	double SOR_LIM=1e-7;
	
	for (ii = 1; ii <= niter; ii++){
		sum = 0.0;
		sdif = 0.0;
		
		// Loop over 2-D aperture
		for (i = 1; i < Nmirror_actmap[0]-1; i++){
			for (j = 1; j < Nmirror_actmap[1]-1; j++){
				//if (mirror_apt(i,j) > 0) {
					// pixel is within membrane boundary
	
					// Calculate update value
					update = -mirror_resp(i, j) - 
							(mirror_actmap(i,j) -
							mirror_resp(i-1, j) - 
							mirror_resp(i+1, j) - 
							mirror_resp(i,   j-1) - 
							mirror_resp(i,   j+1))/4.;
					mirror_resp(i,j) = mirror_resp(i,j) + omega * update;
					sum += mirror_resp(i,j);
					sdif += (omega * update) * (omega * update);
				//} else {
				//	mirror_resp(i,j) = 0.0;
				//}
			}
		} // end of loop over aperture
		
		sdif = sqrt(sdif/(sum*sum));
		if (sdif < SOR_LIM)	{ 
			// stop if changes are small enough
			return_val = ii;
			break;
		}
	}
			
	return_val = ii;
	"""
	return_val = sp.weave.inline(poisson_solver, \
		['mirror_resp', 'mirror_actmap', 'rho', 'omega', 'niter', 'sdif'], \
		extra_compile_args= [__COMPILE_OPTS], \
		type_converters=sp.weave.converters.blitz)

	mirror_resp *= mirror_apt
	
	if (verb > 2): print "dmmodel.sim_dm(): niter=%d, final sdif=%g" % (return_val, sdif)
	
	cropmask = (slice(None), slice(None))
	if (docrop):
		# Find regions where mirror_apt is 0 on all rows or columns:
		apt_max0 = mirror_apt.max(axis=0)
		crop0 = slice(np.argwhere(apt_max0 > 0)[0], np.argwhere(apt_max0 > 0)[-1]+1, 1)
		apt_max1 = mirror_apt.max(axis=1)
		crop1 = slice(np.argwhere(apt_max1 > 0)[0], np.argwhere(apt_max1 > 0)[-1]+1, 1)
		cropmask = (crop0, crop1)
		if (verb > 2): print "dmmodel.sim_dm(): cropmask:", cropmask
	
	return mirror_resp[cropmask]

### EOF
