#!/usr/bin/env python
# encoding: utf-8
"""
@file fit.py
@brief Fit routines and convenience functions

@package libtim.fit
@brief Fit routines and convenience functions
@author Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
@copyright Creative Commons Attribution-Share Alike license versions 3.0 or higher, see http://creativecommons.org/licenses/by-sa/3.0/
@date 20130323

Some routines to fit functions to data, get error estimates and generally 
convenience functions to help me remember.
"""

#============================================================================
# Import libraries here
#============================================================================

from libtim import shell
import scipy as sp
import scipy.optimize

#============================================================================
# Defines
#============================================================================

#============================================================================
# Routines
#============================================================================

def fit_gauss(x, y, guess=None, offset=True):
	"""
	Fit data **x**, **y** to a Gaussian function plus a constant offset if **offset** is True.

	Gauss function fitted is given by:

		a * \exp( -(x - b)^2 / (2*c^2) ) + d
	
	where **d** is only used if **offset** is True.

	Normal distributions with width σ and expected value µ compare to this 
	function as:

		a = 1/(σ√(2π)), b = μ, c = σ

	To calculate the FWHM of a Gaussian, use

		FWHM =  2 sqrt(2 ln 2) * c ~ 2.35482 * c

	Alternatively, the parameter c can be interpreted by saying that the two 
	inflection points of the function occur at x = b − c and x = b + c.

	@param x X-data
	@param y Y-data
	@param offset Add constant offset to fit, i.e. Gauss() + a
	@param guess Guess parameters as starting point, as (a, b, c, [d]) defaults to 0
	@return Tuple of (amplitude, x-offset, sigma, y-offset)
	"""

	if (offset):
		gauss = lambda x, a, b, c, d: d + a*np.exp( -(x - b)**2.0 / (2*c**2))
	else:
		gauss = lambda x, a, b, c: a*np.exp( -(x - b)**2.0 / (2*c**2) )
		
	# Future: use sech as well
	#sech = lambda x, a, b, c, d: d + (a/np.cosh((x-b)/c))**2.0

	if (not guess):
		guess = [0]*(3+offset)

	fitres = sp.optimize.curve_fit(gauss, x, y, p0=guess[:])

	return (fitres[0], np.diag(fitres[1])**0.5)

