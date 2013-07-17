#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""
@file 1394cm.py Camera wrapper routines using cv
@author Tim van Werkhoven
@date 20130717
@copyright Copyright (c) 2013 Tim van Werkhoven (werkhoven@strw.leidenuniv.nl)
"""

import file
import im
import cv
import os
import numpy as np
from os.path import join as pjoin

# Global camera configuration dict
# @TODO Make OO
CAM_CFG = {}
CAM_FLATFIELD="cam_flatfield.fits"
CAM_DARKFIELD="cam_darkfield.fits"
CAM_APTMASK="cam_apt_mask.fits"

# Verbosity masks
VERB_M = (1<<4)-1

# Verb levels, use as: (verb&VERB_M) > L_INFO
L_INFO = 0; L_XNFO = 1; L_DEBG = 2; L_XDBG = 3

def cam_setup(camidx=0, roi=None, flatf=None, darkf=None, maskshape='all', procd=32, usecam=True, outdir='./', verb=0):
	"""
	Setup IEEE1394 camera using Python's binding for OpenCV. Setup will be 
	stored in global CAM_CFG dictionary for use by cam_getimage().

	@param camidx Camera index
	@param roi Region of Interest to use, as (x, y, width, height)
	@param flatf Flatfield image to use [file]
	@param darkf Darkfield image to use [file]
	@param maskshape Shape for processing mask, 'circ' or 'all'
	@param procd Bitdepth to process images at (32 or 64)
	@param outdir Directory to store stuff to
	@param verb Verbosity
	"""

	global CAM_CFG

	if (verb&VERB_M > L_INFO):  print "Setting up camera..."

	if (procd == 64): 
		npdtype = np.float64
		cvdtype = cv.IPL_DEPTH_64F
	else: 
		npdtype = np.float32
		cvdtype = cv.IPL_DEPTH_32F

	if (darkf and os.path.isfile(darkf)):
		if (verb&VERB_M > L_DEBG): print "Processing dark frames..."
		# Hard-link to used files for new cache
		newdarkf = pjoin(outdir, CAM_DARKFIELD)
		if (os.path.exists(newdarkf)):
			os.unlink(newdarkf)
		os.link(darkf, newdarkf)
		darkim = file.read_file(darkf).astype(npdtype)
		CAM_CFG['dark'] = cv.fromarray(darkim)
	if (flatf and os.path.isfile(flatf)):
		if (verb&VERB_M > L_DEBG): print "Processing flat frame(s)..."
		newflatf = pjoin(outdir, CAM_FLATFIELD)
		if (os.path.exists(newflatf)):
			os.unlink(newflatf)
		os.link(flatf, newflatf)
		flatim = file.read_file(flatf).astype(npdtype)
		CAM_CFG['flat'] = cv.fromarray(flatim)
  		if (CAM_CFG.has_key('dark')):
  			cv.Sub(CAM_CFG['flat'], CAM_CFG['dark'], CAM_CFG['flat'])

	if (verb&VERB_M > L_XNFO):  print "Configuring camera..."

	if (not CAM_CFG.has_key('window')):
		CAM_CFG['window'] = 'cam_live'
	cv.NamedWindow(CAM_CFG['window'], cv.CV_WINDOW_AUTOSIZE)
	cv.NamedWindow("cam_histogram", cv.CV_WINDOW_AUTOSIZE)
	CAM_CFG['idx'] = camidx
	CAM_CFG['roi'] = roi
	
	if (usecam): 
		CAM_CFG['handle'] = cv.CaptureFromCAM(camidx)
		#cv.GetCaptureProperty(CAM_CFG['handle'], cv.CV_CAP_PROP_FPS)
		cv.SetCaptureProperty(CAM_CFG['handle'], cv.CV_CAP_PROP_FPS, 60)
	else:
		CAM_CFG['handle'] = None

	if (roi):
		CAM_CFG['dshape'] = (roi[2], roi[3])
	elif (usecam):
		# GetSize returns (width, h), NumPy arrays expect (height, w)
		rawframe = cv.QueryFrame(CAM_CFG['handle'])
		CAM_CFG['dshape'] = cv.GetSize(rawframe)[::-1]
	else:
		raise ValueError("Need ROI or camera to determine data shape.")

	CAM_CFG['frame'] = cv.CreateImage(CAM_CFG['dshape'][::-1], cvdtype, 1)

	if (maskshape == 'circ'):
		CAM_CFG['mask'] = im.mk_rad_mask(*CAM_CFG['dshape']) < 1
	else:
		CAM_CFG['mask'] = np.ones(CAM_CFG['dshape']).astype(np.bool)
	CAM_CFG['imask'] = (CAM_CFG['mask'] == False)

	file.store_file(pjoin(outdir, CAM_APTMASK), CAM_CFG['mask'].astype(np.uint8), clobber=True)

	if (verb&VERB_M > L_INFO):  print "Camera setup complete..."

	cam_getimage(show=True)

camhist, camhistimg = None, None
def cam_getimage(show=False, dfcorr=True, raw=False, showhisto=True, waitkey=25):
	"""
	Get image from the camera, convert, scale, dark-flat correct,
	optionally show this and return as numpy.ndarray.

	If **raw* is set, return the (scaled/ROI'd) image as CvImage

	If CAM_CFG['flat'] or CAM_CFG['dark'] are set, use these to dark-flat 
	correct the image.

	@param [in] show Show image after acquisition
	@param [in] dfcorr Do dark-flat correction
	@param [in] raw Return raw IplImage (scaled and ROI'd, w/o DF correction)
	@param [in] showhisto Show histogram as well (only with **show**)
	@param [in] waitkey Wait time for cv.WaitKey() If 0, don't call. (only with **show**)
	@return Image data as numpy.ndarray
	"""

	if (not CAM_CFG['handle']): return

	rawframe = cv.CloneImage(cv.QueryFrame(CAM_CFG['handle']))

	# Downscale color images
	if (rawframe.channels > 1):
		rawsz = cv.GetSize(rawframe)
		if (not CAM_CFG.has_key('rawyuv') or not CAM_CFG['rawyuv']): 
			CAM_CFG['rawyuv'] = cv.CreateImage(rawsz, rawframe.depth, 3)
			CAM_CFG['rawgray'] = cv.CreateImage(rawsz, rawframe.depth, 1)
		cv.CvtColor(rawframe, CAM_CFG['rawyuv'], cv.CV_BGR2YCrCb)
		cv.Split(CAM_CFG['rawyuv'], CAM_CFG['rawgray'], None, None, None)
		rawframe = CAM_CFG['rawgray']

	if (CAM_CFG['roi']):
		rawframe = cv.GetSubRect(rawframe, tuple(CAM_CFG['roi']))

	procf = CAM_CFG['frame']
	cv.ConvertScale(rawframe, procf, scale=1.0/256)

	if (raw):
		return cv.CloneImage(procf)

	if (CAM_CFG.has_key('dark') and dfcorr):
		cv.Sub(procf, CAM_CFG['dark'], procf)
	if (CAM_CFG.has_key('flat') and dfcorr):
		cv.Div(procf, CAM_CFG['flat'], procf)
	# We *don't* apply the aperture mask here because we might need the data
	
	if (show):
		cv.ShowImage(CAM_CFG['window'], procf)
		if (showhisto):
			global camhist, camhistimg
			camhist, camhistimg = calc_1dhisto(procf, hist=camhist, histimg=camhistimg)
			cv.ShowImage("cam_histogram", camhistimg)
		if (waitkey): 
			cv.WaitKey(waitkey)

	depth2dtype = {
		cv.IPL_DEPTH_32F: 'float32',
		cv.IPL_DEPTH_64F: 'float64',
	}

	framearr = np.fromstring(procf.tostring(), 
		dtype=depth2dtype[procf.depth],
		count=procf.width*procf.height*procf.nChannels)
	framearr.shape = (procf.height, procf.width, procf.nChannels)

	return framearr[:, :, 0]

def cam_measurebulk(nframes=100, interactive=True, show=True, norm=False, verb=0):
	"""
	Take **nframes** frames and average these. If **norm** is set, set the 
	average of the summed frame to unity, otherwise it is divided by the 
	number of frames.

	This routine is intended to measure flat and dark frames. Flat frames 
	might be normalized such that dividing by these does not affect the 
	average intensity of the input frame. Dark frames should never be 
	normalized.

	The flatfield is stored in CAM_CFG['flat'] and is used automatically 
	from then on.

	@param [in] nframes Number of frames to average
	@param [in] show Show flat field + one correct image when done
	@param [in] verb Verbosity
	@return Summed and scaled frame.
	"""

	if (verb&VERB_M > L_INFO):
		print "Measuring bulk (n=%d)..." % (nframes)

	if (interactive):
		print "Will measure bulk now, press c to continue..."
		while (True):
			cam_getimage(show=True, waitkey=0)
			if (chr(cv.WaitKey(1) & 255) == "c"):
				print "ok!"
				break

	bulkimg = cam_getimage(show=False, dfcorr=False, raw=True)

	for dummy in xrange(nframes-1):
		cv.Add(bulkimg, cam_getimage(show=False, dfcorr=False, raw=True), bulkimg)

	if (norm):
		cv.ConvertScale(bulkimg, bulkimg, scale=1.0/cv.Avg(bulkimg)[0])
	else:
		cv.ConvertScale(bulkimg, bulkimg, scale=1.0/nframes)

	if (show):
		cv.NamedWindow("cam_bulkimg", cv.CV_WINDOW_AUTOSIZE)
		cv.ShowImage('cam_bulkimg', bulkimg)
		c = cv.WaitKey(20)

	return bulkimg

def cam_convertframe(frame):
	"""
	Convert **frame** to numpy.ndarray
	"""

	depth2dtype = {
		cv.IPL_DEPTH_8U: 'uint8',
		cv.IPL_DEPTH_8S: 'int8',
		cv.IPL_DEPTH_16U: 'uint16',
		cv.IPL_DEPTH_16S: 'int16',
		cv.IPL_DEPTH_32S: 'int32',
		cv.IPL_DEPTH_32F: 'float32',
		cv.IPL_DEPTH_64F: 'float64',
	}

	arr = np.fromstring(
		 frame.tostring(),
		 dtype=depth2dtype[frame.depth],
		 count=frame.width*frame.height*frame.nChannels)

	if (frame.nChannels == 1):
		arr.shape = (frame.height,frame.width)
	else:
		arr.shape = (frame.height,frame.width,frame.nChannels)
	
	return arr

def calc_1dhisto(inframe, nbin=256, scale=2, histh=200, hist=None, histimg=None):
	"""
	Calculate 1D intensity histogram of a iplimage, and return the histogram 
	(as cv2.cv.cvhistogram) and an image representing this histogram (as 
	8bit unsigned iplimage) Use **hist** and **histimg** if they are 
	provided, otherwise create them from scratch.

	To re-use the allocated memory, simply pass the output as input for 
	input **hist** and **histimg**.

	Histogram bar height is calculated as follows:
		
		bin_height = int( bin_count*1.0 / (npix/nbin) * 0.2 * histh )

	where bin_height is in pixels, bin_count is the number of pixels in this 
	bin, npix the total number of pixels in the image, nbin the total bins, 
	such that (npix/nbin) is the average bin count. 0.2 is a factor that 
	sets the average bin height to 20% and histh scales the bin normalized 
	bin height to pixels.

	@param [in] inframe Input frame, as iplimage
	@param [in] nbin Number of intensity bins
	@param [in] scale Histogram image bar width in pixels
	@param [in] histh Histogram image height in pixels
	@param [in] hist Previously allocated cv2.cv.cvhistogram to use
	@param [in] histimg Previously allocated iplimage to use
	@return Tuple of (histogram, histogram image)
	"""

	if (inframe.depth == cv.IPL_DEPTH_32F):
		hranges = [[0, 1]]
	elif (inframe.depth in [cv.IPL_DEPTH_8U, cv.IPL_DEPTH_8S, 
		cv.IPL_DEPTH_16U, cv.IPL_DEPTH_16S, cv.IPL_DEPTH_32S]):
		hranges = None
	else:
		raise ValueError("Unsupported datatype for histogram ('%s')" % (str(inframe.depth)))

	if (not hist):
		hist = cv.CreateHist([nbin], cv.CV_HIST_ARRAY, ranges=[[0, 1]], uniform=1)
	if (not histimg):
		histimg = cv.CreateImage((nbin*scale, histh), cv.IPL_DEPTH_8U, 1)

	cv.CalcHist([cv.GetImage(inframe)], hist)
	#hmin, hmax, __, __ = cv.GetMinMaxHistValue(hist)

	# White noise histogram height should be be 0.2
	npix = np.product(cv.GetDims(inframe))
	histogram = [cv.QueryHistValue_1D(hist, i)*1.0/(npix/nbin) * 0.2 * histh for i in range(nbin)]
	histimg = plot_1dhisto(histogram, scale=scale, histh=histh, histimg=histimg)

	return hist, histimg

def plot_1dhisto(histogram, scale=2, gap=0, histh=200, histimg=None, origin='bottom'):
	"""
	Given histogram data, make a visual represntation in the form of a 
	IplImage, helper function for calc_1dhisto().

	@param scale Width of each bar (pix)
	@param gap Gap between bars (pix)
	@param histh Bar plot height (pix)
	@param origin Origin of histogram bars, 'bottom' (for positive values) or 'center' (for any value)
	@return Graphical cv.iplimage representation of the histogram
	"""

	nbin = len(histogram)
	histh, scale, gap = int(histh), int(scale), int(gap)

	if (not histimg):
		histimg = cv.CreateImage((nbin*scale, histh), cv.IPL_DEPTH_8U, 1)

	cv.Zero(histimg)
	for i, bin_h in enumerate(histogram):
		if (origin == 'bottom'):
			cv.Rectangle(histimg,
				(i*scale, histh),
				((i+1)*scale - 1 - gap, histh-int(bin_h)),
				256, 
				cv.CV_FILLED)
		else:
			cv.Rectangle(histimg,
				(i*scale, histh/2),
				((i+1)*scale - 1 - gap, histh/2-int(bin_h)),
				256, 
				cv.CV_FILLED)

	return histimg