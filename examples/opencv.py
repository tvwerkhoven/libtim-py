#!/usr/bin/env python
# 
# Test OpenCV capture in Python from any available camera

import libtim as tim
import libtim.im
import numpy as np
import cv
from IPython import embed as shell

ibins = 256
scale = 2
histh = 200
# camhist = cv.CreateHist([ibins], cv.CV_HIST_ARRAY, ranges=[[0, 1]], uniform=1)
# camhist_img = cv.CreateImage((ibins*scale, histh), cv.IPL_DEPTH_8U, 1)

camhist = None
camhist_img = None

def calc_1dhisto(inframe, nbin=256, scale=2, histh=200, hist=None, histimg=None):
	"""
	Calculate 1D intensity histogram of a iplimage, and return the histogram 
	itself (as cv2.cv.cvhistogram) and an image representing this histogram (
	as 8bit unsigned iplimage) Use **hist** and **histimg** if they are 
	provided, otherwise create them from scratch.

	To re-use the allocated memory, simply pass the output as input for 
	input **hist** and **histimg**.

	Histogram bar height is calculated as follows:
		
		bin_height = bin_count*1.0/(npix/nbin) * 0.2 * histh

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
		raise ValueError("Unsupported datatype ('%s')" % (str(inframe.depth)))

	if (not hist):
		hist = cv.CreateHist([nbin], cv.CV_HIST_ARRAY, ranges=[[0, 1]], uniform=1)
	if (not histimg):
		histimg = cv.CreateImage((nbin*scale, histh), cv.IPL_DEPTH_8U, 1)

	npix = np.product(cv.GetDims(inframe))
	
	cv.CalcHist([cv.GetImage(inframe)], hist)
	hmin, hmax, _, _  = cv.GetMinMaxHistValue(hist)

	cv.Zero(histimg)
	for i in range(nbin):
		bin_val = cv.QueryHistValue_1D(hist, i)
		bin_h = int( bin_val*1.0/(npix/nbin) * 0.2 * histh )
		cv.Rectangle(histimg,
			(i*scale, histh),
			((i+1)*scale - 1, histh-bin_h),
			256, 
			cv.CV_FILLED)

	return hist, histimg

def repeat():
	global CAM_CFG, lastframe, diffframe, camframe, camhist, camhist_img
	frame = cv.GetSubRect(cv.QueryFrame(CAM_CFG['handler']), CAM_CFG['roi'])

	# This takes 1% CPU:
	#framearr = cv2array(frame)
	# This takes 7% CPU:
	#framearr = framearr.astype(np.float64)*1.0/256
	# This takes 3% CPU:
	cv.ConvertScale(frame, camframe, scale=1.0/256)
	# This takes 2% CPU:
	#camframearr = cv2array(camframe)

	# Calculate (cam-dark)/flat. Without mask might be faster sometimes.
	cv.Sub(camframe, darkframe, camframe, mask=CAM_CFG['mask'])
	cv.Div(camframe, flatframe, camframe, 1)

	# # Calculate cam - last
	cv.Sub(camframe, lastframe, diffframe, mask=CAM_CFG['mask'])

	# Make histogram of camframe
	camhist, camhist_img = calc_1dhisto(camframe, nbin=ibins, scale=scale, histh=histh, hist=camhist, histimg=camhist_img)

	if (LIVE):
		cv.ShowImage("cam_live", camframe)
		cv.ShowImage("cam_other", diffframe)
		cv.ShowImage("cam_histo", camhist_img)
	
		c = cv.WaitKey(10)
		if(c=="n"): #in "n" key is pressed while the popup window is in focus
			pass

	CAM_CFG['buf'][(CAM_CFG['frameidx'])%len(CAM_CFG['buf'])] = camframe
	CAM_CFG['frameidx'] += 1
	lastframe = cv.CloneImage(camframe)

def cv2array(im):
	depth2dtype = {
		cv.IPL_DEPTH_8U: 'uint8',
		cv.IPL_DEPTH_8S: 'int8',
		cv.IPL_DEPTH_16U: 'uint16',
		cv.IPL_DEPTH_16S: 'int16',
		cv.IPL_DEPTH_32S: 'int32',
		cv.IPL_DEPTH_32F: 'float32',
		cv.IPL_DEPTH_64F: 'float64',
	}

	arrdtype=im.depth
	a = np.fromstring(
		 im.tostring(),
		 dtype=depth2dtype[im.depth],
		 count=im.width*im.height*im.nChannels)
	a.shape = (im.height,im.width,im.nChannels)
	return a

def array2cv(a):
	dtype2depth = {
		'uint8':   cv.IPL_DEPTH_8U,
		'int8':    cv.IPL_DEPTH_8S,
		'uint16':  cv.IPL_DEPTH_16U,
		'int16':   cv.IPL_DEPTH_16S,
		'int32':   cv.IPL_DEPTH_32S,
		'float32': cv.IPL_DEPTH_32F,
		'float64': cv.IPL_DEPTH_64F,
	}
	try:
		nChannels = a.shape[2]
	except:
		nChannels = 1
	cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
		  dtype2depth[str(a.dtype)],
		  nChannels)
	cv.SetData(cv_im, a.tostring(),
			 a.dtype.itemsize*nChannels*a.shape[1])
	return cv_im

LIVE=True

imcenter = np.r_[(265, 210)] # (x, y)
rad = 200
# roi = (x, y, width, height)
roi = tuple(imcenter-rad) + tuple([2*rad]*2)

# Make two windows, these are refered to by name
cv.NamedWindow("cam_live", cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow("cam_other", cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow("cam_histo", cv.CV_WINDOW_AUTOSIZE)

CAM_CFG = {'camidx':0, 'frameidx':0, 'handler':None, 'buf': [None]*10, 'roi':roi}

# Initialize camera handler
CAM_CFG['handler'] = cv.CaptureFromCAM(CAM_CFG['camidx'])
rawframe = cv.QueryFrame(CAM_CFG['handler'])
frame = cv.GetSubRect(rawframe, roi)

# Make fake calibration frames from data
darkarr = (np.random.random(cv.GetDims(frame))*10.0/256).astype(np.float32)
darkframe = array2cv(darkarr)

lastframe = cv.CloneImage(darkframe)
camframe = cv.CloneImage(darkframe)
diffframe = cv.CloneImage(darkframe)

# Make real flat field
print "Taking 100 flats..."
frame = cv.GetSubRect(cv.QueryFrame(CAM_CFG['handler']), CAM_CFG['roi'])
cv.ConvertScale(frame, camframe, scale=1.0/256)
flatframe = cv.CloneImage(camframe)

for i in xrange(9):
	print ".",
	frame = cv.GetSubRect(cv.QueryFrame(CAM_CFG['handler']), CAM_CFG['roi'])
	cv.ConvertScale(frame, camframe, scale=1.0/256)
	cv.Add(flatframe, camframe, flatframe)

cv.ConvertScale(flatframe, flatframe, scale=1.0/cv.Avg(flatframe)[0])

# flatarr = np.linspace(2.0, 0.5, darkarr.shape[0]).reshape(-1,1)
# flatarr = np.dot(flatarr, np.ones((1, darkarr.shape[1])))
# flatframe = array2cv(flatarr)


# Get new frame. QueryFrame returns a pointer to the internal data. We 
# immediately clone the frame because we want to modify it. 
cv.ConvertScale(frame, lastframe, scale=1.0/256)
cv.ShowImage("cam_live", lastframe)

# lastarr = cv2array(lastframe)
# framearr = cv2array(frame)

mask = np.zeros((2*rad, 2*rad), dtype=np.uint8)
mask[:] = True
mask[:rad] = False

#mask = (tim.im.mk_rad_mask(2*rad, norm=True, center=None) < 1).astype(np.uint8)

shell()

CAM_CFG['mask'] = array2cv(mask)
CAM_CFG['buf'][0] = lastframe
CAM_CFG['frameidx'] += 1


while True:
	repeat()
