import os
import pathlib
import torch
from SiameseNetwork import SiameseNetwork
from DataLoader     import  CustomDataSet
from torch.utils.data        import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

def start_test():

    #os.mkdir('results/')

    # Dataset Path
    path_input= '../../../../../DATA/laura/tcia_temp/test/'
    path      = Path(path_input)
    filenames = list(path.glob('*.npy'))

    # Testing dataset
    test_dataset = CustomDataSet(path_dataset = filenames,
                                  img_size    = 256,
                                  transform   = None)

    # Training dataloader
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        torch.device('cpu')

    # criterion
    criterion = torch.nn.BCELoss()

    # model
    model         = SiameseNetwork(backbone='resnet50').to(device)
    model_name    = 'net.pt'
    model_weights = torch.load(pathlib.Path.cwd() / model_name)
    model.load_state_dict(model_weights)
    model.eval()

    inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                              std  = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                         transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                              std  = [ 1., 1., 1. ]),
                                         ])
    # Losses
    losses  = []
    correct = 0
    total   = 0

    for i, (x_1, x_2, y) in enumerate(test_dataloader):
        # send to device (GPU or CPU)
        img_1 = x_1.to(device)
        img_2 = x_2.to(device)
        y = y.to(device)

        with torch.no_grad():
            out         = model(img_1, img_2)
            loss        = criterion(out, y)
            loss_value  = loss.item()
            losses.append(loss_value)
            correct += torch.count_nonzero(y == (out > 0.5)).item()
            total += len(y)

            fig = plt.figure("class1={}\tclass2={}".format(y, y), figsize=(4, 2))
            plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(y, out[0][0].item(), y))

            # Apply inverse transform (denormalization) on the images to retrieve original images.
            img_1 = inv_transform(img_1).cpu().numpy()[0]
            img_2 = inv_transform(img_2).cpu().numpy()[0]
            # show first image
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(img_1[0], cmap=plt.cm.gray)
            plt.axis("off")

            # show the second image
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(img_2[0], cmap=plt.cm.gray)
            plt.axis("off")

            # show the plot
            plt.savefig(os.path.join('results/', '{}.png').format(i))

    print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))

start_test()