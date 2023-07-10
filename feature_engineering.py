""" LIBRARIES """
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy import ndimage

import sklearn 
from sklearn import preprocessing
from skimage import color, exposure, filters, img_as_float, img_as_uint
from skimage.filters import rank
from skimage.morphology import disk

import time
import cv2

#######################################

#convert a folder of images into a target folder
NUM_CHANNELS = 9


def imshow(im):
	#cv2.imwrite("img.png", im)
	plt.imshow(im)
	plt.show()


def imshow_sidebyside(im1, im2):
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
	ax1.imshow(im1, cmap=plt.cm.gray)
	ax1.axis('off')
	ax1.set_title('one', fontsize=20)
	ax2.imshow(im2, cmap=plt.cm.gray)
	ax2.axis('off')
	ax2.set_title('two', fontsize=20)
	plt.show()


def process_channel(im, bil_mode, bil_footprint, mdl_footprint):
	#imshow(im)
	cl = exposure.equalize_adapthist(image=im, kernel_size=10, clip_limit=0.001, nbins=125)
	#imshow(cl)
	if bil_mode == 'r1':
		bil = rank.mean_bilateral(image=cl, footprint=bil_footprint, s0=100, s1=10)
	elif bil_mode == 'r2':
		bil = rank.mean_bilateral(image=cl, footprint=bil_footprint, s0=10, s1=100)
	elif bil_mode == 'normal':
		bil = rank.mean_bilateral(image=cl, footprint=bil_footprint, s0=25, s1=25)
	# imshow(bil)
	mdl = rank.modal(image=bil, footprint=mdl_footprint)
	# imshow(mdl)
	edg_mdl = filters.scharr(mdl)
	# imshow(edg_mdl)
	edg_mcl = exposure.equalize_adapthist(image=edg_mdl, kernel_size=10, clip_limit=0.001, nbins=125)
	# imshow(edg_mcl)
	df0_edg = pd.DataFrame(edg_mcl)
	df0_edg['saved_idx'] = df0_edg.index
	df1_edg = pd.melt(frame=df0_edg, id_vars=['saved_idx'])
	df2_edg = pd.DataFrame(df1_edg.value.values.astype(float).reshape(-1, 1))
	df3_edg = df2_edg.replace(0, np.NaN)
	q3 = np.nanpercentile(a=df3_edg, q=75)
	df2_edg[df2_edg > q3] = q3

	min_max_scaler = preprocessing.MinMaxScaler()
	edg_scaled = min_max_scaler.fit_transform(df3_edg)
	df1_edg['scaled'] = edg_scaled
	df1_edg = df1_edg[['variable', 'saved_idx', 'scaled']]
	edges = df1_edg.pivot(index='saved_idx', columns='variable', values='scaled')
	df4_edg = edges.replace(np.NaN, 0)
	ed_np = np.array(df4_edg, dtype=float)
	# imshow(ed_np)
	return ed_np


def main():
	img_rgb = cv2.imread('AM195_Left_NsNn_V1_Section0001_Probe0001_001.tif')

	imshow(img_rgb)

	img_hsv = color.convert_colorspace(arr=img_rgb, fromspace='rgb', tospace='hsv')

	""" Segregate uncorrelated channels and perform gaussian filtering for lower-frequency feature arrays """
	input_im_arr = []
	# Red/value channel
	input_im_arr.append(img_rgb[:, :, 0])
	input_im_arr.append(ndimage.gaussian_filter(input=input_im_arr[0], sigma=10, order=0, mode='nearest'))
	input_im_arr.append(ndimage.gaussian_filter(input=input_im_arr[0], sigma=40, order=0, mode='nearest'))
	# Hue channel
	input_im_arr.append(img_hsv[:, :, 1])
	input_im_arr.append(ndimage.gaussian_filter(input=input_im_arr[0], sigma=10, order=0, mode='nearest'))
	input_im_arr.append(ndimage.gaussian_filter(input=input_im_arr[0], sigma=40, order=0, mode='nearest'))
	# Saturation channel
	input_im_arr.append(img_hsv[:, :, 2])
	input_im_arr.append(ndimage.gaussian_filter(input=input_im_arr[3], sigma=10, order=0, mode='nearest'))
	input_im_arr.append(ndimage.gaussian_filter(input=input_im_arr[3], sigma=40, order=0, mode='nearest'))

	im_arr = []
	params = [
		('r1', disk(10), disk(20)),
		('r1', disk(30), disk(30)),
		('r1', disk(50), disk(50)),
		('normal', disk(10), disk(20)),
		('normal', disk(30), disk(30)),
		('normal', disk(50), disk(50)),
		('normal', disk(10), disk(10)),
		('normal', disk(30), disk(30)),
		('normal', disk(50), disk(50)),
	]
	for i in range(NUM_CHANNELS):
		im_arr.append(process_channel(input_im_arr[i], params[i][0], params[i][1], params[i][2]))

	imshow(im_arr[0])


if __name__ == '__main__':
	main()

