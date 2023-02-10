import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


"""
Function for weights initialization
"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        m.weight.data = nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find("Linear") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def draw_curve(train_losses, valid_losses):
    x = list(np.arange(len(train_losses)))
    plt.plot(x, train_losses)
    plt.plot(x, valid_losses)
    plt.legend(['Train', 'Valid'])
    plt.savefig('train_error.png')

    print('Image saved!')