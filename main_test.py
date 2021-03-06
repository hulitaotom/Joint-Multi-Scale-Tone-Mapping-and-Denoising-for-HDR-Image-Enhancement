import numpy as np
import cv2
import os
import glob
import argparse
import torch
import matplotlib.pyplot as plt
import time
import torch_dct as dct
from models import *
import sys
import rawpy


def read_raw(path, ratio=1):
	raw = rawpy.imread(path)
	im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)*ratio
	im = cv2.cvtColor(np.float32(im / 65535.0), cv2.COLOR_RGB2BGR) # convert from RGB to BGR
	# returning an BGR image!
	return im

def centeredCosineWindow(x, windowSize=16):
	'''1D version of the modified raised cosine window (Section 4.4 of the IPOL article).
	It is centered and nonzero at x=0 and x=windowSize-1'''
	y = 1 / 2 - 1 / 2 * np.cos(2 * np.pi * (x + 1 / 2.) / windowSize)
	return y

def enhance(model, inp, out_path):
	PATCH_SIZE = 224
	with torch.no_grad():
		height, width, _ = inp.shape
		inp = np.pad(inp, ((PATCH_SIZE,PATCH_SIZE), (PATCH_SIZE,PATCH_SIZE), (0,0)), 'edge')
		inp = torch.from_numpy(inp).float()
		inp = inp.permute(2,0,1).unsqueeze(0)
		if torch.cuda.is_available():
			inp = inp.cuda()
		output = np.zeros((inp.shape[2], inp.shape[3], 3))

		lineWeights = centeredCosineWindow(np.arange(PATCH_SIZE), PATCH_SIZE).reshape(-1, 1).repeat(PATCH_SIZE, 1)
		columnWeights = lineWeights.T
		# the 2D window is the product of the 1D window in both patches dimensions
		window = np.multiply(lineWeights, columnWeights)

		start_time = time.time()
		for r in range(0, inp.shape[2]-PATCH_SIZE, PATCH_SIZE//2):
			for c in range(0, inp.shape[3]-PATCH_SIZE, PATCH_SIZE//2):
				patch = inp[:,:,r:r+PATCH_SIZE, c:c+PATCH_SIZE]
				output_patch = model(patch)
				output_patch = output_patch.permute(0,2,3,1).squeeze().detach().cpu().numpy()
				output[r:r+PATCH_SIZE, c:c+PATCH_SIZE,:] += output_patch*window[:,:,None]
		total_time = time.time() - start_time

	cv2.imwrite(out_path, np.uint8(np.clip(output[PATCH_SIZE:height+PATCH_SIZE,PATCH_SIZE:width+PATCH_SIZE,:], 0.0, 1.0)*255))
	return total_time


def process(config):
	if config.model_type == 'TFDL':
		model = TFDL()
		model_path = "./snapshots/TFDL.pth"
	elif config.model_type == 'DFTL':
		print("This model type is not yet ready.")
		return
		model = DFTL()
		model_path = "./snapshots/DFTL.pth"
	else:
		print("Incorrect model type, please double check input arguments.")
		return 
	if torch.cuda.is_available():
		model = model.cuda()
		model.load_state_dict(torch.load(model_path))
	else:
		model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()

	if not os.path.exists(config.output_folder):
		os.makedirs(config.output_folder)

	output_folder = os.path.join(config.output_folder, config.model_type)
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	paths = glob.glob(os.path.join(config.input_folder, '*'))
	for inp_path in paths:
		if inp_path.split('.')[-1] == 'dng':
			inp_img = read_raw(inp_path)
		else:
			inp_img = cv2.imread(inp_path).astype(float)/255.0
		inp_name = os.path.basename(inp_path).split('.')[0]
		out_path = os.path.join(output_folder, inp_name+'.png')
		time = enhance(model, inp_img, out_path)

	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--input_folder', type=str, default="./examples")
	parser.add_argument('--model_type', type=str, default="TFDL")
	parser.add_argument('--output_folder', type=str, default="./results")


	config = parser.parse_args()

	with torch.no_grad():

		if not os.path.exists(config.output_folder):
			os.makedirs(config.output_folder)

		process(config)