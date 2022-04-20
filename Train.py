import numpy as np
from utils.metrics import *

class Train:
    def __init__(self,
                 model                  : torch.nn.Module,
                 device                 : torch.device,
                 criterion              : torch.nn.Module,
                 optimizer              : torch.optim.Optimizer,
                 training_DataLoader    : torch.utils.data.Dataset,
                 validation_DataLoader  : torch.utils.data.Dataset = None,
                 lr_scheduler           : torch.optim.lr_scheduler = None,
                 epochs                 : int  = 200,
                 epoch                  : int  = 0,
                 notebook               : bool = False
                 ):

        self.model                 = model
        self.criterion             = criterion
        self.optimizer             = optimizer
        self.lr_scheduler          = lr_scheduler
        self.training_DataLoader   = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device                = device
        self.epochs                = epochs
        self.epoch                 = epoch
        self.notebook              = notebook

        self.training_loss         = []
        self.validation_loss       = []
        self.learning_rate         = []

        print('My device is: ', self.device)

    def run_train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            # Epoch counter
            self.epoch += 1

            # Training block
            self._train()

            # Validation block
            if self.validation_DataLoader is not None:
                self._validate()

            # Learning rate scheduler block
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])   # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []   # accumulate the losses here
        batch_iter   = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                            leave=False)

        for i, (x_1, x_2) in batch_iter:

            # send to device (GPU or CPU)
            fixed  = x_1.to(self.device) # Fixed
            moving = x_2.to(self.device) # Moving

            # zero-grad the parameters
            self.optimizer.zero_grad()

            # one forward pass
            t_0, w_0, t_1, w_1 = self.model(fixed, moving)

            # Compute affine network loss
            cc_affine_loss            = pearson_correlation(fixed, w_0)
            det_aff_loss, ort_aff_loss= affine_loss(t_0)

            # Compute deformation network loss
            cc_elastic_loss      = pearson_correlation(fixed, w_1)
            total_variation_loss = elastic_loss_3D(t_1)

            print("cc_affine_loss: ",       cc_affine_loss.item())
            print("det_aff_loss: ",         det_aff_loss.item())
            print("ort_aff_loss: ",         ort_aff_loss.item())
            print("cc_elastic_loss: ",      cc_elastic_loss.item())
            print("total_variation_loss: ", total_variation_loss.item())



            # PAM loss
            loss          = cc_affine_loss + 0.1 * (det_aff_loss + ort_aff_loss) + cc_elastic_loss + 0.1 * total_variation_loss
            loss_value    = loss.item()
            train_losses.append(loss_value)

            # one backward pass
            loss.backward()

            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            # update the parameters
            self.optimizer.step()

            # update progressbar
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')
            print(f'------------------------------- Training: (loss {loss_value:.4f})')
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter   = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                            leave=False)

        for i, (x, y) in batch_iter:

            # send to device (GPU or CPU)
            fixed  = x.to(self.device) # Fixed
            moving = y.to(self.device) # Moving

            with torch.no_grad():
                t_0, w_0, t_1, w_1 = self.model(fixed, moving)

                # Compute affine network loss
                cc_affine_loss        = pearson_correlation(fixed, w_0)
                det_aff_loss, ort_aff_loss = affine_loss(t_0)

                # Compute deformation network loss
                cc_elastic_loss      = pearson_correlation(fixed, w_1)
                total_variation_loss = elastic_loss_3D(t_1)

                # PAM loss
                loss       = cc_affine_loss + 0.1 * (det_aff_loss + ort_aff_loss)  + cc_elastic_loss + 0.1 * total_variation_loss
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
                print(f'------------------------------- Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()
