# TextBoxes: A Fast Detector with a Single Deep Neural Network, in MXNet
A [MXNet](https://beta.mxnet.io/index.html) implementation of [TextBoxes]( https://arxiv.org/pdf/1611.06779.pdf) from the 2016 paper by Minghui Liao, Baoguang Shi, Xiang Bai, Xinggang Wang, Wenyu Liu.  The official and original Caffe code can be found [here]( https://github.com/MhLiao/TextBoxes).


### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#training-textboxes'>Train</a>
- <a href='#todo'>Future Work</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [MXNet](https://beta.mxnet.io/index.html) by selecting your environment on the website and running the appropriate command.
- Install [GluonCV](https://gluon-cv.mxnet.io/) 
- Clone this repository.

## Training TextBoxes

- To train TextBoxes using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.
- Currently training is supported using LST file (See gluon-cv website for examples)

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## TODO
I have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [ ] Upload pretrained models 
  * [ ] Adjust output to ICDAR format
  * [ ] Tensorboard Support

## Authors

* [**Yaniv Azar**](https://github.com/yazar1993)

***Note:*** Unfortunately, this is just a hobby of mine and not a full-time job, so I’ll do my best to keep things up to date, but no guarantees.  That being said, thanks to everyone for your continued help and feedback as it is really appreciated. I will try to address everything as soon as possible.

