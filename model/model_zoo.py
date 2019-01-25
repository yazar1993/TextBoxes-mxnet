# pylint: disable=wildcard-import, unused-wildcard-import
"""Model store which handles pretrained models from both
mxnet.gluon.model_zoo.vision and gluoncv.models
"""
from .textboxes import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'textboxes_300_vgg16_atrous_custom' : textboxes_300_vgg16_atrous_custom,
    'textboxes_512_vgg16_atrous_custom': textboxes_512_vgg16_atrous_custom,
    'textboxes_512_resnet18_v1_custom': textboxes_512_resnet18_v1_custom,
    'textboxes_512_resnet50_v1_custom': textboxes_512_resnet50_v1_custom,
    'textboxes_512_mobilenet1.0_custom': textboxes_512_mobilenet1_0_custom,
    }

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net, base_size = _models[name](**kwargs)
    return net, base_size

def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()
