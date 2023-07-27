""" LIBRARIES """
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil

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
def get_img_array(img, channel_name):
	# for GBR Rrepresented images and respective chanels
	if channel_name == "red":
		return img[:, :, 2]
	if channel_name == "blue":
		return img[:, :, 1]
	if channel_name == "green":
		return img[:, :, 0]
	
	#converting GBR images to HSV to get their respective channels
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	if channel_name  == "hue":
		return hsv_img[:, :, 0]
	if channel_name  == "saturation":
		return hsv_img[:, :, 1]
	if channel_name  == "value":
		return hsv_img[:, :, 2]
	
	return img[:, :, 2]
def convert_images(input_image_folder, output_image_folder, input_image_type, output_image_type, output_channels):
	#remove the output folder if it exists and create a new one
	if os.path.exists(output_image_folder):
		shutil.rmtree(output_image_folder)
	os.mkdir(output_image_folder)
	
	#traverse through each image in the input image folder
	for img_file in os.listdir(input_image_folder):

		image_path = os.path.join(input_image_folder, img_file)
		image = cv2.imread(image_path)

		#for every channel feature calcualtion will be added to this list
		channels = []
		for channel in output_channels:
			channel_wrapper = 'normal'
			if ":" in channel:
				channel_name, feature_name = channel.split(":")

				if channel_name == 'red' : channel_wrapper = 'r1'
				feature_img = process_channel(get_img_array(image, channel_name), channel_wrapper, disk(10), disk(20))
				channels.append(feature_img)
			else:
				if channel == 'red' : channel_wrapper = 'r1'
				feature_img = process_channel(get_img_array(image, channel), channel_wrapper, disk(10), disk(20))
				channels.append(feature_img)
		
		#get the input image file name without the extension 
		input_file_name = os.path.splitext(img_file)[0]
		i = 0
		for img_channels in channels:
			#use f string to name the output file correctly with respect to its channel
			output_name = f"{input_file_name}_{i:04}.{output_image_type}"
			output_pth = os.path.join(output_image_folder, output_name)
			cv2.imwrite(output_pth, img_channels)
			i += 1

#-----------
NUM_CHANNELS = 9


def imshow(im):
	#cv2.imwrite("img.png", im)
	im = im[:, :, 0] * 3
	im = np.array(im, dtype=np.uint8)
	cv2.imwrite("AM195_Left_NsNn_V1_Section0001_Probe0001_001.tif", im)
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
	#testing converted_images function
	input_image_folder= 'input_image_files_GD'
	output_image_folder = 'output_imag_files_GD'
	input_image_type = 'tif'
	output_image_type = 'tif'
	output_channels = ['red', 'green', 'blue', 'hue', 'saturation', 'value', 'red:feature1']
	convert_images(input_image_folder, output_image_folder, input_image_type, output_image_type, output_channels)


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

