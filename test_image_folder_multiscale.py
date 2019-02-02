from matplotlib import pyplot as plt
import time
import numpy as np
import mxnet as mx
from gluoncv.utils import viz
from mxnet import gpu
from mxnet.ndarray import concat
from model import  model_zoo
import glob
from os.path import basename
import gluoncv as gcv

classes = ['text']
rootPath ='/ICDAR2013_test/' 
netName = 'textboxes_512_mobilenet1.0_custom'
path_to_model = 'textBoxes_512_mobilenet1.0_text_ICDAR_new.params'
gpu_ind = 0
output_path = '/results_icdar_test/'
file_ext = '.jpg'

net, _ = model_zoo.get_model(netName, classes=classes, pretrained_base=False)
net.reset_class(classes)
net.load_parameters(path_to_model)

ctx = [mx.gpu(gpu_ind)]
net.collect_params().reset_ctx(ctx)

imagePaths = glob.glob(rootPath + "*" + file_ext)
start = time.time()
scales = (300,500,700,1000)
for imageIdx,imagePath in enumerate(imagePaths):
	bbox_overall = []
	score_overall =[]
	cid_overall = []
	print(imagePath)
	for scale_idx,scale in enumerate(scales):
		x, image = gcv.data.transforms.presets.ssd.load_test(imagePath,scale,max_size=2048)
		y = x.copyto(mx.gpu(gpu_ind))
		print(np.shape(y))
		cid, score, bbox = net(y)
		if scale_idx == 0:
			bbox_overall = bbox[0]
			score_overall = score[0]
			cid_overall = cid[0]
		else:
			bbox_overall = concat(bbox_overall, bbox[0], dim=0)
			score_overall = concat(score_overall, score[0], dim=0)
			cid_overall = concat(cid_overall, cid[0], dim=0)
	ax = viz.plot_bbox(image, bbox_overall, score_overall, cid_overall, class_names=classes)
	base_image_name = basename(imagePath)
	plt.savefig(output_path + base_image_name )
print(time.time()-start)

