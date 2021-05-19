'''





parameters for ACGAN image:
D
# filters = [16, 32, 64, 128, 256, 512]
# strides = [2,1,2,1,2,1]
G
#g_filters = [384, 192, 96, 48, 3]
#g_strids = [1, 2, 2, 2]

#g_filters = [512, 256, 128, 64]
#g_strides = [1, 2, 2, 2]
#g_filters = [512, 256, 128, 3]
#g_filters = [512, 256, 128, 1]

'''


import torch
import sys
import os
import numpy as np

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from CNN.resnet import ResNet18
import torch.backends.cudnn as cudnn
from attacks.cw import CW
from attacks.FGSM import FGSM
import argparse
from load_data import load_data, UnNormalize, Normalize
#from GAN.cgan import cDCGAN

from GAN.cgan_im import cDCGAN
#from GAN.dcgan import DCGAN
from GAN.acgan_1 import ACGAN
import torchvision
import random


data = 'MNIST'

train = False

if data=='cifar10':

    training_set, test_set = load_data(data='cifar10')
#training_set, test_set = load_data(data='mnist')
    model = ResNet18()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.load_state_dict(torch.load('./model_weights/resnet_cifar10.pth'))
    im_channel = 3

if data=='MNIST':
    training_set, test_set = load_data(data='mnist')
    model = ResNet18(dim=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.load_state_dict(torch.load('./model_weights/resnet_mnist.pth')['net'])
    im_channel = 1


model.eval()
model.cuda()

dataloader = torch.utils.data.DataLoader(training_set, batch_size=100, shuffle=True, num_workers=2)

in_dim = 50    # todo change back
class_dim = 10



d_filters = [16, 32, 64, 128, 256, 512]
d_strides = [2, 1, 2, 1, 2, 1]

g_filters = [384, 192, 96, 48, im_channel]
g_strides = [1, 2, 2, 2]

#gan = cDCGAN(in_dim=in_dim, class_dim=class_dim, g_filters=g_filters,
#               g_strides=g_strides, CNN=model)
gan = ACGAN(in_dim=in_dim, class_dim=class_dim, g_filters=g_filters,
            g_strides=g_strides, d_filters=d_filters, d_strides=d_strides, CNN=model)
testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=2)
#gan.train(dataloader, num_epochs=50, test_loader=testloader)

gan.load_model('check_point')

gan.generator.eval()
gan.discriminator.eval()

newloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)




cw = CW(model=model, targeted=1)
fgsm = FGSM(model=model)

for img, ta in newloader:
    break


clean_loss = []

attack_loss = []

def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])




total = 0
correct = 0
norm = 0
for i, (images, labels) in enumerate(newloader):
    target_label = get_target(labels)
    images, labels = images.cuda(), labels.cuda()



    outputs = model(images)
    _, predicted = outputs.max(1)
    if predicted.item() != labels.item():
        continue

    total += 1

    _, loss = gan.reconstruct_loss(images, labels)
    clean_loss.append(loss)

    #adv_images = cw.attack(images, target_label)
    adv_images = fgsm.attack(images, target_label)
    _, aux_out = gan.discriminator(adv_images)
    _, aux_pred = aux_out.max(1)
    if aux_pred.item() == labels.item():
        correct += 1

    _, loss = gan.reconstruct_loss(adv_images, target_label)
    attack_loss.append(loss)

    print(correct, total)



    if i >= 400:
        break

root = 'mnist_fgsm_conf'
os.makedirs(root, exist_ok=True)


np.save(root+'/c_l.npy', np.array(clean_loss))

np.save(root+'/a_l.npy', np.array(attack_loss))








