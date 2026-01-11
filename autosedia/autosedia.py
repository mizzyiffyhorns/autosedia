# This software was created as a general purpose automatic image
# analysis tool and is currently being deveopled. 
# At this moment, it's sole function is to overlay 
# two images, segment the features, and caculate the area overlap.
# See the images included in 'Images' directory for an example
# of the type of images this script will work on.

"""
Auto-processor for dual images.\n
How to run script:
Place your images in the 'Images' directory.
Make sure the file format is as follows: mask01.tiff, target01.tiff
for one set, mask02.tiff, target02.tiff for the second set, etc...
use option -I #_OF_IMAGES for total number of images you want 
to run and analyze at one time. See --help for more options.
"""

__version__ = "0.1.0"

__author__ = ["Mizzy"]


import os
from pathlib import Path
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy.fft as fft
from scipy.ndimage import fourier_gaussian, laplace, distance_transform_edt, maximum_filter
import pandas as pd
from skimage.measure import regionprops_table, label, regionprops
from skimage import morphology
import cv2
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border, watershed
from skimage.morphology import area_closing
from skimage.feature import peak_local_max
from scipy import signal
import torch
from scipy import ndimage as ndi # for remove_large_objects function
from datetime import datetime # for naming outputfiles directory with current time/date


def fourier_gauss(img, sigma, pad=True, pad_width=50):
	""" 
	Computes the fourier gaussian of an image.
	"""
	if pad:
		img = np.pad(img.copy(), pad_width=int(pad_width), mode="edge")
	image_Input = fft.fft2(img)
	result = fourier_gaussian(image_Input, sigma=sigma)
	result = fft.ifft2(result)
	img = result.real
	if pad:
		img = img[pad_width:-pad_width, pad_width:-pad_width]
	return img


def _watershed(thresh_img, directory):
	"""
	Implements the watershed function from scikit-image
	"""
	image = thresh_img
	distance = distance_transform_edt(image)
	coords = peak_local_max(distance, footprint=np.ones((9, 9)), labels=image) # change footprint as needed

	mask = np.zeros(distance.shape, dtype=bool)
	mask[tuple(coords.T)] = True
	markers = label(mask)
	watered = watershed(-distance, markers, mask=image)

	return watered


def remove_large_objects(ar, max_size=500, connectivity=1, *, out=None):
# This function was borrowed from scipy morphology's remove_small_objects function and then modified
    """
	Remove objects larger than the specified size.
    """
	
    if out is None:
        out = ar.copy()
    else:
       out[:] = ar

    if max_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out



def no_holes_vec(closed, intensity_image):
	"""
	Custom vectorized process for filling holes left in foreground pixels
	"""
	closed = closed.copy()
	closed_label = closed.copy()

	# Make an initial labeled array to get average areas of labels to minimize dilation effects
	temp_label = label(closed_label)
	labeldata = pd.DataFrame(regionprops_table(label_image=temp_label,
		intensity_image=intensity_image, properties=('label', 'area', 'mean_intensity', 'weighted_centroid')))

	label_mean = np.mean(labeldata['area'])

	cp = None
	se = None
	if label_mean > 0 and label_mean < 85:
		se = 3
		cp = int((se-1)/2)

	elif label_mean >= 85 and label_mean <= 150:
		se = 9
		cp = int((se-1)/2)

	elif label_mean > 150:
		se = 13
		cp = int((se-1)/2)

	# Padding of array
	closed_pad = np.pad(closed, [cp,cp], mode='constant', constant_values=0)

	# Create the structuring element
	struc_element = np.full((se, se), 1)

	# Creat the Window
	input = closed_pad
	kernel_size = struc_element.shape[0]
	layer_stride = 1

	height, width = input.shape
	rows_stride, columns_strides = input.strides

	output_height = int((height-kernel_size)/layer_stride + 1)
	output_width = int((width-kernel_size)/layer_stride + 1)

	new_shape = (output_height, output_width, kernel_size, kernel_size)
	new_strides = (rows_stride * layer_stride, columns_strides * layer_stride, rows_stride, columns_strides)

	windowed_input = np.lib.stride_tricks.as_strided(input, new_shape, new_strides)

	windowed_input = torch.Tensor(windowed_input)
	struc_element = torch.Tensor(struc_element)

	multiplied = torch.mul(windowed_input,struc_element)
	summed = torch.sum(multiplied, dim=(-1,2))

	summed=summed.numpy()

	# make binary
	summed[summed <= 0] = 0
	summed[summed > 0] = 1

	return summed


def img_label(thresh1, intensity_image, directory):
	"""
	The labeling process - closes, fills holes, watersheds, then labels
	the features, followed by removing small objects, then the large objects.
	"""
	img_binary = thresh1.copy()
	img_binary[img_binary <= 0] = 0 
	img_binary[img_binary > 0] = 1

	# closing 
	closed = area_closing(img_binary, 8, connectivity=1)

	fill_holes = no_holes_vec(closed, intensity_image)

	# convert dtype maskoat32 to int32 for watershedding
	fill_holes = fill_holes.astype(np.int32)

	# watershedding
	water = _watershed(fill_holes, directory)

	img_label = label(water)
	img_label = morphology.remove_small_objects(img_label, 5)
	img_label = remove_large_objects(img_label)

	return img_label, img_binary, fill_holes


def bg(array, sigma, niter=2, typ="soft"):
	"""
	Performs background estimation
	"""
	background = array.copy()
	for i in range(niter):
		background = 1 / fourier_gauss(1 / background, sigma, pad_width=int(4 * sigma))
		background[array < background] = array[array < background]
	return background


def combined_histo(igg_expressed_data, igg_unexpressed_data, marker_expressed_data, marker_unexpressed_data,
			num_igg, num_marker, marker_name):
	"""
	creates final histogram from IgG and Marker Positive
	"""

	#mu2 = np.mean(unexpressed_target.target_log_intensity.values)
	minimum_value = min(min(marker_expressed_data),min(marker_unexpressed_data))
	maximum_value = max(max(marker_expressed_data),max(marker_unexpressed_data))
	minimum_value = min(min(igg_expressed_data), min(igg_unexpressed_data), min(marker_expressed_data),min(marker_unexpressed_data))
	maximum_value = max(max(igg_expressed_data), max(igg_unexpressed_data), max(marker_expressed_data),max(marker_unexpressed_data))
	num_total = num_igg + num_marker
		
	num_bins = np.linspace(math.ceil(minimum_value), math.ceil(maximum_value), 
			round(num_total))#/10)*10) # round number objects to nearths tens and right interval up to nearest int

	#num_bins = np.linspace(1,10,50)

	fig, ax = plt.subplots(figsize=(15,10))

	# Histogram of the data
	n, bins, patches = ax.hist((marker_expressed_data), num_bins,alpha=.8,color='red', density=False,label= marker_name)
	n2, bins2, patches2 = ax.hist((marker_unexpressed_data), num_bins,alpha=.8,color='red', density=False)
	n3, bins3, patches3 = ax.hist((igg_expressed_data), num_bins,alpha=.4,color='black', density=False,label= 'IgG')
	n4, bins4, patches4 = ax.hist((igg_unexpressed_data), num_bins,alpha=.4,color='black', density=False)


	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

	ax.set_xlabel("Log Intensity",fontsize=35, labelpad=20)
	ax.set_ylabel("Density",fontsize=35,labelpad=20)
	ax.legend(fontsize=25)

	plt.xticks(fontsize=30)
	plt.yticks(fontsize=30)
	plt.minorticks_on()

	# Tweak spacing to prevent clipping of ylabel
	plt.tight_layout()
	plt.close()

	return fig

class SVT():

	def mask(self, mask_raw, bgs, sigma, directory, idd):
		# Background estimation
		mask_bg = bg(mask_raw, bgs)
		# SBR image
		imgmaskSBR = mask_raw/mask_bg
		# Laplacian of the Gaussian(LoG)
		imgFgauss = fourier_gauss(imgmaskSBR, sigma)
		imgLaplace = laplace(imgFgauss)
		# Adaptive thresholding 
		eightbit = cv2.normalize(imgLaplace, None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U)
		constantC = 3
		thresh1 = cv2.adaptiveThreshold(eightbit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
										  cv2.THRESH_BINARY_INV, 201,constantC)

		thresh1 = clear_border(thresh1)
		labeled, img_binary, fill_holes = img_label(thresh1, mask_raw, directory)

		# Creating plots 
		px = 1/plt.rcParams['figure.dpi']  # pixel in inches
		fig, ax = plt.subplots(figsize=(1920*px, 1460*px))
		ax.imshow(mask_raw, cmap='gray') #fig dpi = 96
		for region in regionprops(labeled):
		#take regions with large enough areas
			if region.area <= 20000:
				#draw rectangle around segmented blobs
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
										fill=False, edgecolor='red', linewidth=2)
				ax.add_patch(rect)

		ax.set_axis_off()
		plt.tight_layout()
		plt.savefig(directory+'/mask0'+str(idd)+'_Overlay.png')
		plt.close()
		return labeled, img_binary, imgmaskSBR

	def target(self, target_raw, bgs, sigma, directory, idd):
		# Background estimation
		target_bg = bg(target_raw, bgs)
		# SBR image
		imgtargetSBR = target_raw/target_bg
		# Laplacian of the Gaussian(LoG)
		targetimgFgauss = fourier_gauss(imgtargetSBR, sigma)
		targetimgLaplace = laplace(targetimgFgauss)
		# Adaptive thresholding
		targeteightbit = cv2.normalize(targetimgLaplace, None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U)

		targetthresh1 = cv2.adaptiveThreshold(targeteightbit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		                                 cv2.THRESH_BINARY_INV, 201,5)


		targetthresh1 = clear_border(targetthresh1)

		targetlabeled, img_binary, fill_holes = img_label(targetthresh1, target_raw, directory)

		# Creating plots
		px = 1/plt.rcParams['figure.dpi']  # pixel in inches
		fig, ax = plt.subplots(figsize=(1920*px, 1460*px))
		ax.imshow(target_raw, cmap='gray') #fig dpi = 96

		for region in regionprops(targetlabeled):
		#take regions with large enough areas
			if region.area <= 20000:
				#draw rectangle around segmented blobs
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
										fill=False, edgecolor='yellow', linewidth=2)
				ax.add_patch(rect)

		ax.set_axis_off()
		plt.tight_layout()
		plt.savefig(directory+'/target0'+str(idd)+'_Overlay.png')
		plt.close()
		return targetlabeled, img_binary, imgtargetSBR


	def overlay(self, target_raw, masklabeled, targetlabeled, directory, idd):
		#Overlay of mask and target labels
		px = 1/plt.rcParams['figure.dpi']  # pixel in inches
		fig, ax = plt.subplots(figsize=(1920*px, 1460*px))
		ax.imshow(target_raw, cmap='gray') #fig dpi = 96

		for region in regionprops(masklabeled):
			# take regions with large enough areas
			if region.area <= 8000:
				# draw rectangle around segmented blobs
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
									  fill=False, edgecolor='red', linewidth=2)
				ax.add_patch(rect)

		for region in regionprops(targetlabeled):
			# take regions with large enough areas
			if region.area <= 8000:
				# draw rectangle around segmented blobs
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
									  fill=False, edgecolor='yellow', linewidth=2)
				ax.add_patch(rect)	

		ax.set_axis_off()
		plt.tight_layout()
		plt.savefig(directory+'/Pair0'+str(idd)+'_Overlay.png')
		plt.close()

	def data(self, mask_raw, mask_SBR, target_raw, target_SBR, labelmask, labeltarget, img_binary, directory, idd, thresh = 0.05):
		# Create datafiles
		maskdata = pd.DataFrame(regionprops_table(label_image=labelmask,
					intensity_image=mask_raw, properties=('label', 'area', 'mean_intensity', 'weighted_centroid')))

		maskdata['SBR'] = regionprops_table(label_image=labelmask,
					intensity_image=mask_SBR, properties=('label', 'mean_intensity'))["mean_intensity"]

		targetdata = pd.DataFrame(regionprops_table(label_image=labeltarget,
					intensity_image=target_raw, properties=('label', 'area', 'mean_intensity', 'weighted_centroid')))

		targetdata['SBR'] = regionprops_table(label_image=labeltarget,
					intensity_image=target_SBR, properties=('label', 'mean_intensity'))["mean_intensity"]


		data = pd.DataFrame(regionprops_table(label_image=labelmask,
					intensity_image=mask_raw, properties=('label', 'area', 'mean_intensity', 'weighted_centroid')))

		data["target_intensity"] = regionprops_table(label_image=labelmask,
					intensity_image=target_raw, properties=('label', 'mean_intensity'))["mean_intensity"]

		data["target_log_intensity"] = np.log(data.target_intensity.values)

		data['SBR'] = regionprops_table(label_image=labelmask,
					intensity_image=target_SBR, properties=('label', 'mean_intensity'))["mean_intensity"]

		data["size_fraction"] = regionprops_table(label_image=labelmask,
					intensity_image=img_binary, properties=('label', 'mean_intensity'))["mean_intensity"]

		data["expressed"] = data.size_fraction.values > thresh
		expressed_sum = sum((data['expressed'] == True))

		expressed_target = pd.DataFrame(data.loc[data['expressed'] == True])
		unexpressed_target = pd.DataFrame(data.loc[data['expressed'] == False])

		#mu1 = np.mean(expressed_target.target_log_intensity.values)
		mu2 = np.mean(unexpressed_target.target_log_intensity.values)
		min_value = data['target_log_intensity'].min()
		expressed_target["normalized_target_log_intensity"] = (expressed_target["target_log_intensity"] / min_value) -1
		unexpressed_target["normalized_target_log_intensity"] = (unexpressed_target["target_log_intensity"] / mu2) -1

		maskdata.to_csv(directory + "/maskdata0"+str(idd)+".csv")
		targetdata.to_csv(directory + "/targetdata0"+str(idd)+".csv")
		data.to_csv(directory + "/data0"+str(idd)+".csv")

		self.data = data
		self.maskdata = maskdata
		self.targetdata = targetdata
		self.expressed_sum = expressed_sum
		self.expressed_target = expressed_target
		self.unexpressed_target = unexpressed_target

		return self.data, self.maskdata, self.targetdata, self.expressed_sum, self.expressed_target, self.unexpressed_target


	def histo(self, expressed_target, unexpressed_target, num_markers, directory, idd):
		# Create histograms
		if idd == 10:
			idd = 'Final'

		#sigma = np.std(expressed_target)
		#mu = np.mean(expressed_target.target_log_intensity.values)

		#sigma2 = (np.std(unexpressed_target))
		mu2 = np.mean(unexpressed_target.target_log_intensity.values)
		
		num_bins = np.linspace(0, math.ceil(max(max(expressed_target['normalized_target_log_intensity']),max(unexpressed_target['normalized_target_log_intensity']))), 
				round(num_markers))#/10)*10) # round number objects to nearths tens and right interval up to nearest int

		#num_bins = np.linspace(1,10,50)

		fig, ax = plt.subplots(figsize=(15,10))

		# Histogram of the data
		n, bins, patches = ax.hist((expressed_target['normalized_target_log_intensity']), num_bins, density=False)

		n2, bins2, patches2 = ax.hist((unexpressed_target['normalized_target_log_intensity']), num_bins, density=False, alpha = 0.5)


		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		ax.set_xlabel("Log Intensity",fontsize=35, labelpad=20)
		ax.set_ylabel("Density",fontsize=35,labelpad=20)

		plt.xticks(fontsize=30)
		plt.yticks(fontsize=30)
		plt.minorticks_on()

		# Tweak spacing to prevent clipping of ylabel
		plt.tight_layout()
		fig.savefig(directory + '/histo' + str(idd) + '.jpg')
		plt.close()


def main():
	import argparse
	from argparse import RawTextHelpFormatter

	SCRIPT_DIR = os.environ.get("AUTOSEDIA_SCRIPT_DIR")
	if SCRIPT_DIR:
		SCRIPT_DIR = str(Path(SCRIPT_DIR))
	else:
		SCRIPT_DIR = str(Path.cwd())

	#directory = os.getcwd()
	directory = SCRIPT_DIR
	current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
	raw_image_directory = directory+'/data/RawImages'
	pre_image_directory = directory+'/data/ManualBinary'
	num_raw_images = int(len([entry for entry in os.listdir(raw_image_directory) if os.path.isfile(os.path.join(raw_image_directory, entry))])/2) # counts number of images
	num_pre_images = int(len([entry for entry in os.listdir(pre_image_directory) if os.path.isfile(os.path.join(pre_image_directory, entry))])/2) # counts number of images
	output_parentDirectory = directory+'/data/OutputFiles/'
	output_childDirectory = directory+'/data/OutputFiles/'+str(current_datetime)

	ap = argparse.ArgumentParser(
		description=__doc__, formatter_class=RawTextHelpFormatter
	)

	ap.add_argument(
		"-IM",
		"--img_dir",
		default=raw_image_directory,
		help="Directory where images are located",
		)
	ap.add_argument(
		"-V",
		"--version",
		action="version",
		version="%(prog)s {}".format(__version__),
		help="Prints the version, then exits script",
		)
	ap.add_argument(
		"-DS",
		"--dsigma",
		type=int,
		default=5,
		help="Specifies sigma value for dark-field fourier-gaussian filter - default=5"
		)
	ap.add_argument(
		"-FS",
		"--fsigma",
		type=int,
		default=5,
		help="Specifies sigma value for fluorescence fourier-gaussian filter - default=5"
		)
	ap.add_argument(
		"-DB",
		"--targetbackground",
		type=int,
		default=5,
		help="Specifies sigma value for dark-field background - default=5"
		)
	ap.add_argument(
		"-FB",
		"--maskbackground",
		type=int,
		default=5,
		help="Specifies sigma value for fluorescence background - default=5"
		)
	ap.add_argument(
		"-I",
		"--images",
		type=int,
		default=num_raw_images,
		help="Number of image PAIRS to analyze - default=1"
		)
	ap.add_argument(
		"-M",
		"--manual",
		action='store_true',
		help="Enter each argument manually for each image pair"
		)
	ap.add_argument(
		"-P",
		"--preprocessed",
		action='store_true',
		help="run preprocessed images instead of raw images"
		)
	ap.add_argument(
		"-H",
		"--histo",
		action='store_true',
		help="run histogram creation only"
		)

	cmd = ap.parse_args()

	results_complete = None
	expressed_target_combined = None
	unexpressed_target_combined = None
	data_complete = None

	if cmd.preprocessed == True:
		try:
			from modules.ManualSVT import ManualSVT
			os.mkdir(output_childDirectory)
		except FileExistsError:
			print("\n The OutputFiles directory already exists! Please move it so overwritting",
			"does not occur\n")
			sys.exit(1)
		if num_pre_images >= 1:
			print("\n[+] Analysis Starting...\n")
			for i in range(cmd.images):
				output_directory = output_childDirectory+'/ImageSet'+str(i+1)
				os.mkdir(output_directory)

				idd = i + 1 

				image_params = [cmd.maskbackground, cmd.targetbackground, cmd.fsigma, cmd.dsigma]
				image_params = pd.DataFrame(np.array(image_params)).T
				image_params.columns = ['mask Background', 'target Background', 'mask Sigma', 'target Sigma']
				image_params.to_csv(output_directory + "/image_params0"+str(idd)+".csv")

				svt = ManualSVT()
				mask_image = np.asarray(Image.open(raw_image_directory+'/mask0' + str(idd) + '.tif'))
				mask_binary = np.asarray(Image.open(pre_image_directory+'/maskbinary0' + str(idd) + '.tif'))
				labelmask, img_binarymask = svt.mask(mask_image, mask_binary, cmd.maskbackground, cmd.fsigma, output_directory, idd)

				target_image = np.asarray(Image.open(raw_image_directory+'/target0' + str(idd) + '.tif'))
				target_binary = np.asarray(Image.open(pre_image_directory+'/targetbinary0' + str(idd) + '.tif'))
				labeltarget, img_binarytarget = svt.target(target_image, target_binary, cmd.targetbackground, cmd.dsigma, output_directory, idd)

				svt.overlay(target_image, labelmask, labeltarget, output_directory, idd)
				svt.data(mask_image, target_image, labelmask, labeltarget, img_binarytarget, output_directory, idd)

				n_vesicles = svt.maskdata.shape[0]
				n_markers = svt.targetdata.shape[0]
				expressed_total = svt.expressed_sum
				frac_positive = expressed_total/n_vesicles
				n_false_markers = n_markers - expressed_total

				hist = svt.histo(svt.expressed_target, svt.unexpressed_target, n_vesicles, output_directory, idd)

				results = pd.DataFrame(np.array([n_vesicles, n_markers, expressed_total, frac_positive, n_false_markers])).T
				results.columns = ['n_vesicles', 'n_markers', 'expressed_total', '%total_positive', 'n_false_markers']

				# final data
				if isinstance(data_complete, pd.DataFrame):
					data_complete = pd.concat([data_complete, svt.data])
				else:
					data_complete = svt.data
				# final results
				if isinstance(results_complete, pd.DataFrame):
					results_complete = pd.concat([results_complete, results])
				else:
					results_complete = results

				# final histogram expressed_target data
				if isinstance(expressed_target_combined, pd.DataFrame):
					expressed_target_combined = pd.concat([expressed_target_combined, svt.expressed_target])
				else:
					expressed_target_combined = svt.expressed_target

				# final histogram unexpressed_target data
				if isinstance(unexpressed_target_combined, pd.DataFrame):
					unexpressed_target_combined = pd.concat([unexpressed_target_combined, svt.unexpressed_target])
				else:
					unexpressed_target_combined = svt.unexpressed_target

				print('\n[+] Image Set #'+str(idd), 'done.\n')

			results_complete = results_complete.reset_index()
			results_complete.to_csv(output_directory + "/results.csv")

			total_expressed = sum(results_complete['expressed_total'].values)\
											/sum(results_complete['n_vesicles'].values)*100

			final_hist = svt.histo(expressed_target_combined, unexpressed_target_combined, 
											sum(results_complete['n_vesicles'].values),  output_directory, 10)

			data_complete.to_csv(output_directory + "/data_combined.csv")
			expressed_target_combined.to_csv(output_directory + "/expressed_target_combined.csv")
			unexpressed_target_combined.to_csv(output_directory + "/unexpressed_target_combined.csv")


			print("\n[+] Results:")
			print(f"	Total # of vesicles = {sum(results_complete['n_vesicles'].values)}\n"
				f"	Total # of markers = {sum(results_complete['n_markers'].values)}\n"
				f"	Total # of marker positive vesicles (expressed) = {sum(results_complete['expressed_total'].values)}\n"
				f"	Total % of expressed vesicles = {round(total_expressed,3)}%\n")
					#run_script = 'python ' + directory + '/modules/labeling_only.py'
					#os.system(run_script)
					#print('Hello')
					#exit()

		else:
			print("\n No Images sets were available in the BinaryImages! \n Place,",
				'at least one image pair in that directory and run the script again. \n')
			exit()

	elif cmd.histo == True:

		histo_output_directory = directory + '/data/histo/OutputFiles/'
		histo_output_childDirectory = directory+ '/data/histo/OutputFiles/' +str(current_datetime)

		try:
			os.mkdir(histo_output_childDirectory)
		except FileExistsError:
			print("\n The OutputFiles directory already exists! Please move it so overwritting",
			"does not occur\n")
			sys.exit(1)
		
		marker_name = input('Please enter the name of the marker that you\'re analyzing then '
			'press enter: ')

		print("\n[+] Creating Histogram...\n")

		igg_directory = directory + '/data/histo/IgG/'
		marker_directory = directory + '/data/histo/MarkerPositive/'
		igg_expressed = pd.read_csv(igg_directory+'expressed_target_combined.csv')
		igg_unexpressed = pd.read_csv(igg_directory+'unexpressed_target_combined.csv')
		marker_expressed = pd.read_csv(marker_directory+'expressed_target_combined.csv')
		marker_unexpressed = pd.read_csv(marker_directory+'unexpressed_target_combined.csv')

		igg_expressed_data = igg_expressed['normalized_target_log_intensity']
		igg_unexpressed_data = igg_unexpressed['normalized_target_log_intensity']
		marker_expressed_data = marker_expressed['normalized_target_log_intensity']
		marker_unexpressed_data = marker_unexpressed['normalized_target_log_intensity']

		num_igg = len(igg_expressed_data) + len(igg_unexpressed_data)
		num_marker = len(marker_expressed_data) + len(marker_unexpressed_data)

		histo = combined_histo(igg_expressed_data, igg_unexpressed_data, marker_expressed_data, marker_unexpressed_data,
			num_igg, num_marker, marker_name)

		histo.savefig(histo_output_childDirectory + '/temp.jpg')

		print("\n[+] Histogram saved!\n")


		exit()


	else:
		if num_raw_images >= 1:
			try:
				os.mkdir(output_childDirectory)
			except FileExistsError:
				print("\n The OutputFiles directory already exists! Please move it so overwritting",
				"does not occur\n")
				sys.exit(1)
			print("\n[+] Analysis Starting...\n")
			for i in range(cmd.images):
				output_directory = output_childDirectory+'/ImageSet'+str(i+1)
				os.mkdir(output_directory)

				idd = i + 1 

				image_params = [cmd.maskbackground, cmd.targetbackground, cmd.fsigma, cmd.dsigma]
				image_params = pd.DataFrame(np.array(image_params)).T
				image_params.columns = ['mask Background', 'target Background', 'mask Sigma', 'target Sigma']
				image_params.to_csv(output_directory + "/image_params0"+str(idd)+".csv")

				svt = SVT()
				mask_image = np.asarray(Image.open(raw_image_directory+'/mask0' + str(idd) + '.tif'))
				labelmask, img_binarymask, mask_SBR = svt.mask(mask_image, cmd.maskbackground, cmd.fsigma, output_directory, idd)

				target_image = np.asarray(Image.open(raw_image_directory+'/target0' + str(idd) + '.tif'))
				labeltarget, img_binarytarget, target_SBR = svt.target(target_image, cmd.targetbackground, cmd.dsigma, output_directory, idd)

				svt.overlay(target_image, labelmask, labeltarget, output_directory, idd)
				svt.data(mask_image, mask_SBR, target_image, target_SBR, labelmask, labeltarget, img_binarytarget, output_directory, idd)

				n_vesicles = svt.maskdata.shape[0]
				n_markers = svt.targetdata.shape[0]
				expressed_total = svt.expressed_sum
				frac_positive = expressed_total/n_vesicles
				n_false_markers = n_markers - expressed_total

				hist = svt.histo(svt.expressed_target, svt.unexpressed_target, n_vesicles,  output_directory, idd)

				results = pd.DataFrame(np.array([n_vesicles, n_markers, expressed_total, frac_positive, n_false_markers])).T
				results.columns = ['n_vesicles', 'n_markers', 'expressed_total', '%total_positive', 'n_false_markers']

				# final data
				if isinstance(data_complete, pd.DataFrame):
					data_complete = pd.concat([data_complete, svt.data])
				else:
					data_complete = svt.data

				# final results
				if isinstance(results_complete, pd.DataFrame):
					results_complete = pd.concat([results_complete, results])
				else:
					results_complete = results

				# final histogram expressed_target data
				if isinstance(expressed_target_combined, pd.DataFrame):
					expressed_target_combined = pd.concat([expressed_target_combined, svt.expressed_target])
				else:
					expressed_target_combined = svt.expressed_target

				# final histogram unexpressed_target data
				if isinstance(unexpressed_target_combined, pd.DataFrame):
					unexpressed_target_combined = pd.concat([unexpressed_target_combined, svt.unexpressed_target])
				else:
					unexpressed_target_combined = svt.unexpressed_target

				print('\n[+] Image Set #'+str(idd), 'done.\n')

			results_complete = results_complete.reset_index()
			results_complete.to_csv(output_directory + "/results.csv")

			total_expressed = sum(results_complete['expressed_total'].values)\
											/sum(results_complete['n_vesicles'].values)*100

			final_hist = svt.histo(expressed_target_combined, unexpressed_target_combined, 
											sum(results_complete['n_vesicles'].values),  output_directory, 10)

			data_complete.to_csv(output_directory + "/data_combined.csv")
			expressed_target_combined.to_csv(output_directory + "/expressed_target_combined.csv")
			unexpressed_target_combined.to_csv(output_directory + "/unexpressed_target_combined.csv")

			print("\n[+] Results:")
			print(f"	Total # of vesicles = {sum(results_complete['n_vesicles'].values)}\n"
				f"	Total # of markers = {sum(results_complete['n_markers'].values)}\n"
				f"	Total # of marker positive vesicles (expressed) = {sum(results_complete['expressed_total'].values)}\n"
				f"	Total % of expressed vesicles = {round(total_expressed,3)}%\n")


if __name__ == '__main__':
	import time
	start = time.perf_counter()
	time.sleep(1)
	main()
	end = time.perf_counter()
	print(f"\n[+] Analysis complete. Main function execution time: {end-start:.03f} seconds",
	   '\n\n[+] See output directory for output files',
	   '\n\n[+] Please report any errors seen while running script to the maintainer!\n')
