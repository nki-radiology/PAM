import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch


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


""" Early Stopping class to stop the training if validation loss doesn't improve after a given patience """
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_temp=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        if model_temp:
            torch.save(model_temp.state_dict(), 'Dis_' + self.path.replace('_earlystopping.pth', '_earlystopping') + '.pth')
        self.val_loss_min = val_loss