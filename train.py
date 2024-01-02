import torch
import torch.nn as nn
from model import build_net
import os
from PIL import Image
from torchvision import transforms
from transunet.vit_seg_modeling import VisionTransformer as TransUNet
from transunet.vit_seg_modeling import CONFIGS as TransUNet_CONFIGS
from optimization import BertAdam
from att_unet import AttU_Net, R2AttU_Net, U_Net, R2U_Net
from deformable_unet.deform_unet import DUNetV1V2 
from nested_unet import NestedUNet
from swinunet.vision_transformer import SwinUnet as ViT_seg
from medt import MedT
from unet_3plus.unet_3plus import UNet_3Plus
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Resize([512,512]),
])

folder_input = 'Training set path'
folder_label = 'Test Set Path'

input_list = os.listdir(folder_input)
label_list = os.listdir(folder_label)

input_list = sorted(input_list)
label_list = sorted(label_list)

input_list = [os.path.join(folder_input, name) for name in input_list]
label_list = [os.path.join(folder_label, name) for name in label_list]

batch_input = []
batch_label = []

for img_path in input_list:
    img = Image.open(img_path)
    img_tensor = transform(img)
    # print(img_tensor.max())
    batch_input.append(img_tensor)

for img_path in label_list:
    img = Image.open(img_path)
    img_tensor = transform(img)
    # print(img_tensor.max())
    batch_label.append(img_tensor)

train_input = torch.stack(batch_input, dim=0)
train_label = torch.stack(batch_label, dim=0)

from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset

batch_size = 2
train_data = MyDataset(train_input, train_label)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

import numpy as np
from torch import optim
from loss import dice_loss
# loss_fn = nn.CrossEntropyLoss()
loss_fn = dice_loss

def get_optimiazer(net):
    return optim.Adam(net.parameters(), lr=0.001)

def print0(*print_args, **kwargs):
    print(*print_args, **kwargs)

import matplotlib.pyplot as plt

def train(train_loader, net, optimizer, loss_fn, num_epochs, device):
    netmax=0
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            if (epoch+1)%5==0 and (i+1)%5==0:
                print('Epoch [{}/{}], Step [{}/{}], Dice coefficient: {:.6f}, Dice Loss: {:.6f}'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), 1-loss.item(), loss.item()))
                '''if(1-loss.item()>netmax):
                    netmax=1-loss.item()
                    torch.save(net, 'net{}.pth'.format(epoch + 1))'''
                correct = 0
                output, label = outputs[-1], labels[-1]
                output = output.permute(1,2,0).detach().cpu().numpy()
                output = np.where(output > 0.5, 1, 0)
                label = label.permute(1,2,0).detach().cpu().numpy()

                imgs = [output, label]

                titles = ['y_pred', 'y_true']

                for i, img in enumerate(imgs):
                    plt.subplot(1, len(imgs), i+1)
                    plt.rcParams["image.cmap"] = "gray"
                    plt.imshow(img)
                    plt.title(titles[i])
                    plt.axis('off')
                plt.show()
                plt.savefig(f'train_pred/fig_epoch_{epoch+1}.png')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
net = build_net()
net.to(device,dtype=torch.float32)
optimizer = get_optimiazer(net)
num_epochs = 100

train(train_loader, net, optimizer, loss_fn, num_epochs, device)

torch.save(net, 'net.pth')