'''
WGAN implementation gan-defense paper
wgan based on paper: https://arxiv.org/abs/1610.09585
gan-defense paper: https://arxiv.org/pdf/1805.06605.pdf
by Hang Wang   hzw81@psu.edu
'''



import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from CNN.resnet import BasicBlock
import torch.nn.functional as F

class WGAN:
    def __init__(self, in_dim, g_filters, g_strides,  feature_size=32):
        # wgan implementation with
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_dim = in_dim
        self.img_channel = g_filters[-1]
        self.g_filters = g_filters
        self.g_strides = g_strides
        self.generator = Generator(self.in_dim, self.g_filters, self.g_strides).to(self.device)
        self.discriminator = Discriminator(self.g_filters[-1],
                                           self.g_filters[::-1][1:], self.g_strides[::-1][:-1]).to(self.device)
        self.weight_clipping_limit = 0.01
        self.critic_iter = 5
        self.feature_size = feature_size


    def load_model(self, dir):
        g_path = 'generator.pth'
        d_path = 'discriminator.pth'
        g_path = os.path.join(dir, g_path)
        d_path = os.path.join(dir, d_path)

        self.generator.load_state_dict(torch.load(g_path))
        self.discriminator.load_state_dict(torch.load(d_path))


    def save_model(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        g_path = 'generator.pth'
        d_path = 'discriminator.pth'
        g_path = os.path.join(dir, g_path)
        d_path = os.path.join(dir, d_path)
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)
    def create_fake_labels(self, labels):
        l = labels.detach().cpu().numpy()
        new_label = []
        for i in range(len(l)):
            a = random.randint(0, self.class_dim-1)
            while a == l[i]:
                a = random.randint(0, self.class_dim - 1)
            new_label.append(a)
        return torch.tensor(new_label).type(torch.LongTensor)



    def train(self, data_loader, lr=0.00005, num_epochs=40):
        G_optimizer = optim.RMSprop(self.generator.parameters(), lr=lr)
        D_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        one = torch.FloatTensor([1]).to(self.device)
        mone = one * -1


        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                ###############################
                # train the discriminator
                ###############################
                self.discriminator.zero_grad()
                batch_size = images.size()[0]
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

                features = images.to(self.device)
                d_loss_real = self.discriminator(features)
                d_loss_real = d_loss_real.mean(0).view(1)
                d_loss_real.backward(one)

                #train the discriminator with fake images
                z = torch.randn([batch_size, self.in_dim]).to(self.device)
                fake_images = self.generator(z)
                d_loss_fake = self.discriminator(fake_images)
                d_loss_fake = d_loss_fake.mean(0).view(1)
                d_loss_fake.backward(mone)
                d_loss = d_loss_fake - d_loss_real


                D_optimizer.step()

                # train G every self.critic_iter step
                if i % self.critic_iter == self.critic_iter-1:
                    for p in self.discriminator.parameters():
                        p.requires_grad = False
                    for p in self.generator.parameters():
                        p.requires_grad = True
                    self.generator.zero_grad()
                    z = torch.randn([batch_size, self.in_dim]).to(self.device)

                    fake_images = self.generator(z)
                    g_loss = self.discriminator(fake_images)
                    g_loss = g_loss.mean(0).view(1)

                    g_cost = -g_loss
                    g_loss.backward(one)
                    G_optimizer.step()
                    for p in self.discriminator.parameters():
                        p.requires_grad = True
                    for p in self.generator.parameters():
                        p.requires_grad = False


                    print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, %.4f, G_loss: %.4f'
                      %(epoch+1, num_epochs, i+1, len(data_loader), d_loss_real.item(), d_loss_fake.item(), g_cost.item()))

    def reconstruct_loss(self, images, lr=0.01, steps=1000):
        batch_size = images.size()[0]
        images = images.to(self.device)
        z = torch.zeros([batch_size, self.in_dim]).to(self.device)
        z.requires_grad = True

        optimizer = optim.Adam([z], lr=lr)
        mse = nn.MSELoss()

        for step in range(steps):
            optimizer.zero_grad()
            g_result = self.generator(z)
            loss = mse(g_result, images)
            loss.backward()
            optimizer.step()

            #print('step: [%d/%d], loss: %.4f' % (step+1, steps, loss.item()))

        return g_result, loss.item()




'''
g_filters = [512, 256, 128, 1]
g_strides = [1, 2, 2, 2]
d_filters = [128, 256, 512]
d_strides = [2, 2, 2]
'''




class Generator(nn.Module):
    def __init__(self, input_dim, filter_nums, strides, kernel_size=4):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.filter_nums = filter_nums
        self.strides = strides
        self.kernel_size = kernel_size
        self.input_seq = nn.Sequential()
        self.hidden_layer = nn.Sequential()
        for i in range(len(self.filter_nums)):
            padding=0 if self.strides[i]==1 else 1
            if i == 0:

                #for input
                self.input_seq.add_module('input_deconv',
                                          nn.ConvTranspose2d(self.input_dim,
                                                             self.filter_nums[i],
                                                             self.kernel_size,
                                                             self.strides[i],
                                                             padding))
                self.input_seq.add_module('input_bn',
                                          nn.BatchNorm2d(self.filter_nums[i], momentum=0.9))
                self.input_seq.add_module('input_act',
                                          nn.LeakyReLU(0.2))
            else:

                deconv_name = 'deconv' + str(i)
                bn_name = 'bn' + str(i)
                act_name = 'act' + str(i)
                self.hidden_layer.add_module(deconv_name,
                                             nn.ConvTranspose2d(self.filter_nums[i-1],
                                                                self.filter_nums[i],
                                                                self.kernel_size,
                                                                self.strides[i],
                                                                padding)
                                             )

                if i == 3:
                    self.hidden_layer.add_module(act_name,
                                                 nn.Tanh())
                else:
                    self.hidden_layer.add_module(bn_name,
                                                 nn.BatchNorm2d(self.filter_nums[i], momentum=0.9))
                    self.hidden_layer.add_module(act_name,
                                                 nn.LeakyReLU(0.2))

    def forward(self, input):
        # Concatenate label embedding and image to produce input
        x = input.view(-1, self.input_dim, 1, 1)
        x = self.input_seq(x)
        x = self.hidden_layer(x)
        return (x + 1)/2



class Discriminator(nn.Module):
    def __init__(self, input_channel, filter_nums, strides, kernel_size=4):
        super(Discriminator, self).__init__()
        self.input_channel = input_channel

        self.filter_nums = filter_nums
        self.strides = strides
        self.kernel_size = kernel_size
        self.label_seq = nn.Sequential()
        self.input_seq = nn.Sequential()
        self.hidden_layer = nn.Sequential()
        self.output_layer = nn.Conv2d(self.filter_nums[-1],
                                      1,
                                      self.kernel_size,
                                      stride=1,
                                      padding=0)

        for i in range(len(self.filter_nums)):
            padding = 0 if self.strides[i] == 1 else 1
            if i == 0:
                # for input
                self.input_seq.add_module('input_conv',
                                          nn.Conv2d(self.input_channel,
                                                    self.filter_nums[i],
                                                    self.kernel_size,
                                                    self.strides[i],
                                                    padding))
                self.input_seq.add_module('input_bn',
                                          nn.BatchNorm2d(self.filter_nums[i], momentum=0.9))
                self.input_seq.add_module('input_act',
                                          nn.LeakyReLU(0.2))
            else:
                conv_name = 'conv' + str(i)
                bn_name = 'bn' + str(i)
                act_name = 'act' + str(i)
                self.hidden_layer.add_module(conv_name,
                                             nn.Conv2d(self.filter_nums[i-1],
                                                       self.filter_nums[i],
                                                       self.kernel_size,
                                                       self.strides[i],
                                                       padding))
                self.hidden_layer.add_module(bn_name,
                                             nn.BatchNorm2d(self.filter_nums[i], momentum=0.9)
                                             )
                self.hidden_layer.add_module(act_name,
                                             nn.LeakyReLU(0.2))


    def forward(self, features, extract_feature=False):

        x = self.input_seq(features)

        x = self.hidden_layer(x)
        out = self.output_layer(x)


        return out.view(-1, 1)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def c_n(out1, out2):
    re = 0
    for i in range(out1.size()[1]):
        re -= out1[0][i]*torch.log(out2[0][i])
    return re



