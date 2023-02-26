import torch
from   torch.utils import data
import numpy as np
from   libs.frida.io         import ImageLoader, ReadVolume
from   libs.frida.transforms import  ZeroOneScaling, ToNumpyArray


class RegistrationDataSet(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 input_shape : tuple,
                 transform   = None
                 ):
        self.dataset     = path_dataset
        self.input_shape = input_shape
        self.indices     = path_dataset.index.values.copy()
        self.transform   = transform
        self.random_seed = int(0)
        self.inp_dtype   = torch.float32
        self.log         = []

        self.loader      = self.__init_loader()

    def __init_loader(self):
        return ImageLoader(
            ReadVolume(),
            ZeroOneScaling(),
            ToNumpyArray(add_batch_dim=False, add_singleton_dim=False)
        )

    def __random_seed(self):
        np.random.seed(self.random.seed)
        self.random_seed += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,
                    index: int):

        # Select the sample
        image_path = self.dataset.iloc[index]
        #print("Image Path: ", image_path)
        fx = np.zeros(self.input_shape)
        mv = np.zeros(self.input_shape)

        fixed_path  = str(image_path.squeeze().dicom_path)
        fx = self.loader(fixed_path)
        fx = fx[:160, :, :]

        fx[:,  :,  0]  = 0
        fx[:,  :, -1]  = 0
        fx[:,  0,  :]  = 0
        fx[:, -1,  :]  = 0
        fx[0,  :,  :]  = 0
        fx[-1,  :,  :] = 0

        moving_path = str(self.dataset.sample(n=1).squeeze().dicom_path)
        mv = self.loader(moving_path)
        mv = mv[:160, :, :]
        mv[:,  :,  0] = 0
        mv[:,  :, -1] = 0
        mv[:,  0,  :] = 0
        mv[:, -1,  :] = 0
        mv[0,  :,  :] = 0
        mv[-1,  :,  :] = 0

        fx = fx.transpose(1, 2, 0)
        fx = torch.from_numpy(fx).type(self.inp_dtype)
        fx = fx[None, :]

        mv = mv.transpose(1, 2, 0)
        mv = torch.from_numpy(mv).type(self.inp_dtype)
        mv = mv[None, :]
        return fx, mv