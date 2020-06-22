
from .fcn import FCN8, FCN16, FCN32
from .unet import UNet
from .fcn_res import RES32, RES16, RES8

__factory = {
    'fcn8': FCN8,
    'fcn16': FCN16,
    'fcn32': FCN32,
    'unet': UNet, 
    'res32': RES32, 
    'res16': RES16,
    'res8': RES8
}


def get_names():
    return __factory.keys()


def get_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
