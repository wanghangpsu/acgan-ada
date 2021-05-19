import torch
from load_data import load_data
from GAN.acgan_res import ACGAN_Res
from CNN.resnet import ResNet18
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
from attacks.cw import CW
import random

training_set, test_set = load_data(data='cifar10')


model = ResNet18()
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

model.load_state_dict(torch.load('./model_weights/resnet_cifar10.pth'))
model.eval()
model.cuda()
dataloader = torch.utils.data.DataLoader(training_set, batch_size=100, shuffle=True, num_workers=2)
in_dim = 500
class_dim = 10


gan = ACGAN_Res(in_dim=in_dim, class_dim=class_dim, CNN=model)



newloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)


#gan.train(dataloader, num_epochs=60, test_loader=newloader)
#gan.save_model('resgan_60')
'''
gan.save_model('resgan_30')
gan.train(dataloader, num_epochs=60, test_loader=newloader)
gan.save_model('resgan_90')
'''
gan.load_model('resgan90')
gan.discriminator.eval()
gan.generator.eval()      #todo those two lines are important!!!
#gan.acc(newloader)

newloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

cw = CW(model=model, targeted=1, c_id_=4)


clean_loss = []


attack_loss = []

correct = 0
total = 0
norm = 0
def get_target(labels):
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])


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

    adv_images =  cw.attack(images, target_label)


    _, loss = gan.reconstruct_loss(adv_images, target_label)
    _, aux_out = gan.discriminator(adv_images)
    _, aux_pred = aux_out.max(1)
    if aux_pred.item() == labels.item():
        correct += 1

    print(correct)


    attack_loss.append(loss)
    print(i)
    if i >= 20:
        break

root = 'cifar_low_conf'


np.save(root+'/c_l.npy', np.array(clean_loss))

np.save(root+'/a_l.npy', np.array(attack_loss))

