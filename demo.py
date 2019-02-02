import time
import numpy as np
import mxnet as mx
from gluoncv.utils import viz
from mxnet import gpu
from mxnet.ndarray import concat
from model import  model_zoo
from os.path import basename
import gluoncv as gcv

classes = ['text']
imagePath ='' 
netName = 'textboxes_512_mobilenet1.0_custom'
path_to_model = 'textBoxes_512_mobilenet1.0_text_ICDAR_new.params'
gpu_ind = 0
output_path = ''

net, _ = model_zoo.get_model(netName, classes=classes, pretrained_base=False)
net.reset_class(classes)
net.load_parameters(path_to_model)

start = time.time()
x, image = gcv.data.transforms.presets.ssd.load_test(imagePath,scale,max_size=2048)
cid, score, bbox = net(y)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
base_image_name = basename(imagePath)
plt.savefig(output_path + base_image_name )
print(time.time()-start)

