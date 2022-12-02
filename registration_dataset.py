import torch
import numpy                  as     np
from   torch.utils            import data
import torchvision.transforms as     transforms
from   PIL                    import Image, ImageOps

class RegistrationDataSet(data.Dataset):
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