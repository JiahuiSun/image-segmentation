
from .fcn import fcn8s, fcn16s, fcn32s, FCN8, FCN16, FCN32
from .unet import unets, UNet

__factory = {
    'fcn16s': fcn16s,
    'fcn8s': fcn8s,
    'fcn32s': fcn32s,
    'fcn8': FCN8,
    'fcn16': FCN16,
    'fcn32': FCN32,
    'unets': unets,
    'unet': UNet
}


def get_names():
    return __factory.keys()


def get_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
