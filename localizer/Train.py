import pathlib

import numpy as np
import torch

class Train:
    def __init__(self,
                 model                  : torch.nn.Module,
                 device                 : torch.device,
                 criterion              : torch.nn.Module,
                 optimizer              : torch.optim.Optimizer,
                 training_DataLoader    : torch.utils.data.Dataset,
                 validation_DataLoader  : torch.utils.data.Dataset = None,
                 lr_scheduler           : torch.optim.lr_scheduler = None,
                 epochs                 : int  = 100,
                 epoch                  : int  = 0,
                 notebook               : bool = False
                 ):

        self.model                  = model
        self.criterion              = criterion
        self.optimizer              = optimizer
        self.lr_scheduler           = lr_scheduler
        self.training_DataLoader    = training_DataLoader
        self.validation_DataLoader  = validation_DataLoader
        self.device                 = device
        self.epochs                 = epochs
        self.epoch                  = epoch
        self.notebook               = notebook

        self.training_loss          = []
        self.validation_loss        = []
        self.learning_rate          = []

    def run_train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    # learning rate scheduler step with validation loss
                    self.lr_scheduler.batch(self.validation_loss[i])
                else:
                    # learning rate scheduler step
                    self.lr_scheduler.batch()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []   # save the losses
        batch_iter   = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                            leave=False)

        for i, (x_1, x_2, y) in batch_iter:
            # send to device (GPU or CPU)
            img_1 = x_1.to(self.device)
            img_2 = x_2.to(self.device)
            label  = y.to(self.device)

            # zero-grad the parameters
            self.optimizer.zero_grad()

            # one forward pass
            out           = self.model(img_1, img_2)

            # compute loss
            loss          = self.criterion(out, label)
            loss_value    = loss.item()
            train_losses.append(loss_value)

            # one backward pass
            loss.backward()

            # update the parameters
            self.optimizer.step()

            # update progressbar
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # save losses here
        batch_iter   = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                            leave=False)

        for i, (x_1, x_2, y) in batch_iter:
            # send to device (GPU or CPU)
            img_1 = x_1.to(self.device)
            img_2 = x_2.to(self.device)
            label  = y.to(self.device)

            with torch.no_grad():
                out         = self.model(img_1, img_2)
                loss        = self.criterion(out, label)
                loss_value  = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()


def plot_training(training_losses,
                  validation_losses,
                  learning_rate,
                  gaussian = True,
                  sigma    = 2,
                  figsize  = (8, 6)
                  ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter

    list_len   = len(training_losses)
    x_range    = list(range(1, list_len + 1))  # number of x values

    fig        = plt.figure(figsize=figsize)
    grid       = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1    = fig.add_subplot(grid[0, 0])
    subfig2    = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss   = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original   = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train   = 'red'
        color_smooth_valid   = 'green'
        alpha                = 0.25
    else:
        linestyle_original   = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    # Subfig 2
    subfig2.plot(x_range, learning_rate, color='black')
    subfig2.title.set_text('Learning rate')
    subfig2.set_xlabel('Epoch')
    subfig2.set_ylabel('LR')

    fig.savefig("error.png")

    return fig


from SiameseNetwork import SiameseNetwork
from DataLoader     import  CustomDataSet
from torch.utils    import data
from torch.utils.data        import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

def start_train():

    # Dataset Path
    path_input= '../../../../../DATA/laura/tcia_temp/train/'
    path      = Path(path_input)
    filenames = list(path.glob('*.npy'))

    # Random seed
    random_seed = 42

    # Split dataset into training set and validation set
    train_size = 0.8
    inputs_train, inputs_valid = train_test_split(
        filenames, random_state=random_seed, train_size=train_size, shuffle=True
    )

    # Training dataset
    train_dataset = CustomDataSet(path_dataset = inputs_train,
                                  img_size     = 256,
                                  transform    = None)

    # Validation dataset
    valid_dataset = CustomDataSet(path_dataset = inputs_valid,
                                  img_size     = 256,
                                  transform    = None)

    # Training dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

    # Validation dataloader
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        torch.device('cpu')

    # model
    model = SiameseNetwork(backbone='resnet50').to(device)

    # criterion
    criterion = torch.nn.BCELoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # trainer
    trainer = Train(model                 = model,
                    device                = device,
                    criterion             = criterion,
                    optimizer             = optimizer,
                    training_DataLoader   = train_dataloader,
                    validation_DataLoader = valid_dataloader,
                    lr_scheduler          = None,
                    epochs                = 100,
                    epoch                 = 0,
                    notebook              = True)

    # start training
    training_losses, validation_losses, lr_rates = trainer.run_train()

    model_name = "net.pt"
    torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)

    return training_losses, validation_losses, lr_rates


train_loss, valid_loss, lr_rates = start_train()
fig = plot_training(
    train_loss,
    valid_loss,
    lr_rates,
    gaussian=True,
    sigma=1,
    figsize=(10, 4),
)
print('End :)')
