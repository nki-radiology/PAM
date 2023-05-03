import torch
import numpy                  as     np
from   torch.utils            import data
import torchvision.transforms as     transforms
from   PIL                    import Image, ImageOps
from   libs.frida.io         import ImageLoader, ReadVolume
from   libs.frida.transforms import  ZeroOneScaling, ToNumpyArray, PadAndCropTo


class Registration2DDataSet(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 input_dim   : int    = [192, 192],
                 transform   : object = None
                ):
        
        self.dataset   = path_dataset
        self.indices   = path_dataset.index.values.copy()
        self.inp_dtype = torch.float32
        
        self.transform = transforms.Compose([
            transforms.Resize(input_dim),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
       # Select the sample
        image_path = self.dataset.iloc[index]
        
        # Fixed Image
        fixed_path  = str(image_path.squeeze().dicom_path)
        fixed       = Image.open(fixed_path)
        fixed       = ImageOps.grayscale(fixed)
        fixed       = self.transform(fixed)
        
        # Moving img
        moving_path = str(self.dataset.sample(n=1).squeeze().dicom_path)
        moving      = Image.open(moving_path)
        moving      = ImageOps.grayscale(moving)
        moving      = self.transform(moving)
        
        return fixed, moving




class Registration3DDataSet(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 input_dim   : int    = [192, 192, 160],
                 transform   : object = None
                 ):
        self.dataset     = path_dataset
        self.input_shape = tuple(input_dim + [1]) # Giving the right shape as (192, 192, 160, 1)
        self.transform   = transform
        self.inp_dtype   = torch.float32
        self.loader      = self.__init_loader()


    def __init_loader(self):
        return ImageLoader(
            ReadVolume(),
            ZeroOneScaling(),
            ToNumpyArray(add_batch_dim=False, add_singleton_dim=False)
        )


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index: int):

        # Select the sample
        #print("Image Path: ", self.dataset.iloc[index])
        
        fx = np.zeros(self.input_shape)
        mv = np.zeros(self.input_shape)

        fixed_path  = str(self.dataset.iloc[index]['PRIOR_PATH_NRRD']) #str(image_path.squeeze().dicom_path)
        #print('Fixed image: ', self.dataset.iloc[index]['PRIOR_PATH_NRRD'])
        fx = self.loader(fixed_path)

        fx[:,  :,  0]  = 0
        fx[:,  :, -1]  = 0
        fx[:,  0,  :]  = 0
        fx[:, -1,  :]  = 0
        fx[0,  :,  :]  = 0
        fx[-1,  :,  :] = 0

        moving_path = str(self.dataset.iloc[index]['SUBSQ_PATH_NRRD']) #str(self.dataset.sample(n=1).squeeze().dicom_path)
        #print('Moving image: ', self.dataset.iloc[index]['SUBSQ_PATH_NRRD'])
        mv = self.loader(moving_path)
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
        
        surv = np.array([self.dataset.iloc[index]['Y1Survival']])
        surv = torch.from_numpy(surv).type(self.inp_dtype)
        return fx, mv, surv