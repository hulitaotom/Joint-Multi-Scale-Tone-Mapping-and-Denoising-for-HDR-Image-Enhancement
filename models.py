import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch_dct as dct
import cv2 as cv
from gaussian_pyramid import *
from csrnet import *

class CNN3(nn.Module):
	def __init__(self, channels=[3*42*42, 3*42*7, 3*7*7, 3*7, 3]):
		super(CNN3, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.e_conv1 = nn.Conv2d(channels[0],channels[1],1,1,0,bias=True)
		self.e_conv2 = nn.Conv2d(channels[1],channels[2],1,1,0,bias=True) 
		self.e_conv3 = nn.Conv2d(channels[2],channels[3],1,1,0,bias=True)
		self.e_conv4 = nn.Conv2d(channels[3],channels[4],1,1,0,bias=True)
	
	def forward(self, x):
		x1 = self.relu((self.e_conv1(x))) #8
		x2 = self.relu((self.e_conv2(x1))) #4
		x3 = self.relu((self.e_conv3(x2))) #2
		y = self.e_conv4(x3) #1
		return y



class denoiseNet_adaptive_thre6(nn.Module):

	def __init__(self, tsize=16, stride=4, in_size=224):
		super(denoiseNet_adaptive_thre6, self).__init__()

		self.L = ((in_size-tsize)//stride+1)**2
		channels = [3*self.L, 3*self.L//4, 3*self.L//16, 3*self.L//4, 3*self.L]
		self.cnn = CNN3(channels)
		self.sigmoid = nn.Sigmoid()
		self.tsize = tsize
		self.stride = stride
		self.unfold = nn.Unfold(tsize, padding=0, stride=stride)
		self.fold = nn.Fold(in_size, tsize, padding=0, stride=stride)
		self.eps = 1e-20
	
	def forward(self, x):
		b,ch,h,w = x.shape
		scale = self.fold(self.unfold(torch.ones(x.shape, requires_grad=False))).cuda()
		tiles = self.unfold(x).permute(0,2,1).reshape(b,-1,self.tsize,self.tsize)
		dct_tiles = dct.dct_2d(tiles)
		masks = self.sigmoid(self.cnn(dct_tiles/(self.tsize*self.tsize)))
		clean_dct_tiles = masks * dct_tiles
		clean_tiles = dct.idct_2d(clean_dct_tiles)
		clean_imgs = self.fold(clean_tiles.reshape(b,-1,ch*self.tsize*self.tsize).permute(0,2,1))
		return clean_imgs / (scale)


class TFDL(nn.Module):
	def __init__(self):
		super(TFDL, self).__init__()
		self.max_lv = 3
		tsize = 16
		stride = 4
		self.scale1 = denoiseNet_adaptive_thre6(tsize, stride, 224)
		self.scale2 = denoiseNet_adaptive_thre6(tsize, stride, 112)
		self.scale3 = denoiseNet_adaptive_thre6(tsize, stride, 56)
		self.enhance = CSRNet()
		self.enhance1 = CSRNet()
		self.enhance2 = CSRNet()
		self.enhance3 = CSRNet()

	def forward(self, x):
		base_in, x3_in, x2_in, x1_in = build_laplacian_pyramid(x, max_level=self.max_lv)
		base = self.enhance(base_in)
		x3 = self.enhance1(x3_in)
		x3 = self.scale3(x3)
		x2 = self.enhance2(x2_in)
		x2 = self.scale2(x2)
		x1 = self.enhance3(x1_in)
		x1 = self.scale1(x1)
		enhanced = reconstruct([base, x3, x2, x1])
		return enhanced


class DFTL(nn.Module):
	def __init__(self):
		super(DFTL, self).__init__()
		self.max_lv = 3
		tsize = 16
		stride = 4
		self.scale1 = denoiseNet_adaptive_thre6(tsize, stride, 224)
		self.scale2 = denoiseNet_adaptive_thre6(tsize, stride, 112)
		self.scale3 = denoiseNet_adaptive_thre6(tsize, stride, 56)
		self.enhance = CSRNet()
		self.enhance1 = CSRNet()
		self.enhance2 = CSRNet()
		self.enhance3 = CSRNet()

	def forward(self, x):
		base_in, x3_in, x2_in, x1_in = build_laplacian_pyramid(x, max_level=self.max_lv)
		base = self.enhance(base_in)
		x3_d = self.scale3(x3_in)
		x3 = self.enhance3(x3_d)
		x2_d = self.scale2(x2_in)
		x2 = self.enhance2(x2_d)
		x1_d = self.scale1(x1_in)
		x1 = self.enhance1(x1_d)
		enhanced = reconstruct([base, x3, x2, x1])
		return enhanced