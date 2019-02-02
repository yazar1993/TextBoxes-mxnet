import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz
from model import model_zoo
import argparse


def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

parser = argparse.ArgumentParser()
parser.add_argument('--images_root',type=str,help='root folder of images')
parser.add_argument('--LSTpath', type=str, help= 'path to LST file')
parser.add_argument('--batch_size', default = 16, type=int)
parser.add_argument('--num_epochs', default = 100, type=int)
parser.add_argument('--lr', type=float, default = 0.001, help='learning rate')
parser.add_argument('--wd', type=float, default = 0.0005)
parser.add_argument('--momentum',type=float,default = 0.9)
parser.add_argument('--netName', type=str, help='name of network to train')
parser.add_argument('--gpu_ind', type=str, help='comma seperated gpu indicies', default = '0')
parser.add_argument('--finetune_model',type=str, help='path to model to finetune from', default = '')
args = parser.parse_args()

images_root = args.images_root
LSTpath = args.LSTpath
classes = ['text'] 
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
wd = args.wd
momentum = args.momentum
netName = args.netName 
gpu_ind=args.gpu_ind
path_to_model = args.finetune_model

# load dataset from Lst file
dataset = gcv.data.LstDetection(LSTpath, root=images_root)
print(dataset)
image= dataset[0][0]
label = dataset[0][1]
print('label:', label)

# display image and label
ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)
plt.savefig('labeled_image.jpg')

#initalize model
net, input_size = model_zoo.get_model(netName, pretrained=False, classes=classes)
if finetune_model == '':
	net.initialize()
	net.reset_class(classes)
else:
	net.load_parameters(path_to_model)
	net.reset_class(classes)
print(net)

train_data = get_dataloader(net, dataset, input_size, batch_size, 0)

#############################################################################################
# Try use GPU for training

try:
    gpu_ind = gpu_ind.split(',')
    ctx = []
    for cur_gpu in gpu_ind:
    	cur_gpu = int(cur_gpu)
    	a = mx.nd.zeros((1,), ctx=mx.gpu(cur_gpu))
    	ctx.append(mx.gpu(cur_gpu))
    print('gpu mode is used')
except:
    print('cpu mode is used')
    ctx = [mx.cpu()]

#############################################################################################
# Start training
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': lr, 'wd': wd, 'momentum': momentum})

mbox_loss = gcv.loss.SSDMultiBoxLoss()
ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')

for epoch in range(0, num_epochs):
    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()
    net.hybridize(static_alloc=True, static_shape=True)
    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            autograd.backward(sum_loss)
        trainer.step(1)
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        if i % 20 == 0:
            print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        btic = time.time()

net.save_parameters(netName + '_icdar2013.params')
