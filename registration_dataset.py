import torch
import numpy                  as     np
from   torch.utils            import data
import torchvision.transforms as     transforms
from   PIL                    import Image, ImageOps
from   libs.frida.io         import ImageLoader, ReadVolume
from   libs.frida.transforms import  ZeroOneScaling, ToNumpyArray


class Registration2DDataSet(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 transform   : object = None):
        self.dataset   = path_dataset
        self.indices   = path_dataset.index.values.copy()
        self.inp_dtype = torch.float32
        self.transform = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
       # Select the sample
        image_path = self.dataset.iloc[index]
        
        fixed_path  = str(image_path.squeeze().dicom_path)
        fixed       = Image.open(fixed_path)
        fixed       = ImageOps.grayscale(fixed)
        fixed       = self.transform(fixed)
        
        #
        moving_path = str(self.dataset.sample(n=1).squeeze().dicom_path)
        moving      = Image.open(moving_path)
        moving      = ImageOps.grayscale(moving)
        moving      = self.transform(moving)
        
        return fixed, moving




class Registration3DDataSet(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 input_shape : tuple = (192, 192, 300),
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
        fx[:,  :,  0]  = 0
        fx[:,  :, -1]  = 0
        fx[:,  0,  :]  = 0
        fx[:, -1,  :]  = 0
        fx[0,  :,  :]  = 0
        fx[-1,  :,  :] = 0

        moving_path = str(self.dataset.sample(n=1).squeeze().dicom_path)
        mv = self.loader(moving_path)
        mv[:,  :,  0] = 0
        mv[:,  :, -1] = 0
        mv[:,  0,  :] = 0
        mv[:, -1,  :] = 0
        mv[0,  :,  :] = 0
        mv[-1, :,  :] = 0

        fx = fx.transpose(1, 2, 0)
        fx = torch.from_numpy(fx).type(self.inp_dtype)
        fx = fx[None, :]

        mv = mv.transpose(1, 2, 0)
        mv = torch.from_numpy(mv).type(self.inp_dtype)
        mv = mv[None, :]
        return fx, mv