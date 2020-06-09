
from .fcn import fcn8s, fcn16s, fcn32s
from .unet import unet

__factory = {
    'fcn16s': fcn16s,
    'fcn8s': fcn8s,
    'fcn32s': fcn32s,
    'unet': unet
}


def get_names():
    return __factory.keys()


def get_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
