"""Import custom conv packages
"""
from .alexnet import AlexNet
from .lenet import LeNet
from .fcheadnet import FCHeadNet
from .minigooglenet import MiniGoogLeNet
from .minivggnet import MiniVGGNet
from .shallownet import ShallowNet

__all__ = [
    'AlexNet',
    'FCHeadNet',
    'LeNet',
    'MiniGoogLeNet',
    'MiniVGGNet',
    'ShallowNet',
    ]
