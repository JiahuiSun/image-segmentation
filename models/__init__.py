
from .fcn import fcn8s, fcn16s, fcn32s, FCN8
from .unet import unets, UNet

__factory = {
    'fcn16s': fcn16s,
    'fcn8s': fcn8s,
    'fcn8': FCN8,
    'fcn32s': fcn32s,
    'unets': unets,
    'unet': UNet
}


def get_names():
    return __factory.keys()


def get_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
