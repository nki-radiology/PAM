import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):

    def __init__(self,
                 backbone: str = 'resnet50'
                 ):
        super().__init__()

        if backbone not in models.__dict__:
            raise Exception('There is no model named {} in torchvision.models.'.format(backbone))

        # Define the backbone network from the pretrained models provided by torchvision models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # Get the features obtained by the last layer of the backbone network
        output_features = list(self.backbone.modules())[-1].out_features

        # MLP to perform a similarity classification
        self.classification = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(output_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img_1, img_2):

        # Get the output features for both images from the backbone
        features_img_1 = self.backbone(img_1)
        features_img_2 = self.backbone(img_2)

        # Element-wise multiplication to create a combined feature vector
        # which represents the similarity between the two images
        com_feature_vec = features_img_1 * features_img_2

        # Get the similarity value
        similarity_value = self.classification(com_feature_vec)

        return similarity_value


# Verifying if the network is working properly
"""
from torchsummary import summary

model = SiameseNetwork( backbone= 'resnet50')
x     = torch.randn(size=(2, 3, 256, 256), dtype=torch.float32)
y     = torch.randn(size=(2, 3, 256, 256), dtype=torch.float32)

with torch.no_grad():
    value = model(x, y)
print(f'value: {value}')

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model   = model.to(device)
summary = summary(model, [(3, 256, 256), (3, 256, 256)], device='cuda')
"""