import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import random

'''
g_filters = [512, 256, 128, 64]
g_strides = [1, 2, 2, 2]
d_filters = [128, 256, 512]
d_strides = [2, 2, 2]
'''
class cDCGAN:
    def __init__(self, in_dim, class_dim, g_filters, g_strides, CNN, feature_id=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_dim = in_dim
        self.class_dim = class_dim
        self.g_filters = g_filters
        self.g_strides = g_strides

        self.generator = Generator(self.in_dim, self.class_dim, self.g_filters, self.g_strides).to(self.device)
        self.discriminator = Discriminator(self.g_filters[-1], self.class_dim,
                                           self.g_filters[::-1][1:], self.g_strides[::-1][:-1]).to(self.device)
        self.CNN = CNN.to(self.device)
        self.feature_id = feature_id

        self.one_hot_table = torch.eye(self.class_dim, self.class_dim)
        self.feature_size = 32
        self.d_input_table = torch.zeros(self.class_dim, self.class_dim, self.feature_size, self.feature_size)
        for i in range(self.class_dim):
            self.d_input_table[i, i, :, :] = 1
        self.d_loss = None
        self.g_loss = None

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



    def train(self, data_loader, lr=0.0002, num_epochs=40):
        d_loss = []
        g_loss = []
        self.CNN.eval()

        bce_loss = nn.BCELoss()
        G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        for epoch in range(num_epochs):
            d_losses = []
            g_losses = []
            if epoch == 10 or epoch == 30:
                G_optimizer.param_groups[0]['lr'] /= 10
                D_optimizer.param_groups[0]['lr'] /= 10

            for i, (images, labels) in enumerate(data_loader):
                # train the discriminator
                self.discriminator.zero_grad()
                batch_size = images.size()[0]
                real_y = torch.ones(batch_size) #+ torch.randn(batch_size) * 0.1
                fake_y = torch.zeros(batch_size) #+ torch.randn(batch_size) * 0.1
                real_y = real_y.to(self.device)
                fake_y = fake_y.to(self.device)
                #extract features from the cnn
                #_, features = self.CNN(images, extract_feature=True)
                #features = features[self.feature_id]
                features = images.to(self.device)
                d_labels = self.d_input_table[labels].to(self.device)
                d_result = self.discriminator(features, d_labels).squeeze()
                d_real_loss = bce_loss(d_result, real_y)

                # real samples with fake label
                #fake_labels = self.create_fake_labels(labels)

                #fake_d_labels = self.d_input_table[fake_labels].to(self.device)
                #fake_d_labels = (1-self.d_input_table)[labels].to(self.device) / 9
                #fake_d_result = self.discriminator(features, fake_d_labels).squeeze()
                #d_fake_label_loss = bce_loss(fake_d_result, fake_y)


                # fake samples
                z = torch.randn([batch_size, self.in_dim, 1, 1]).to(self.device)
                rand_labels = (torch.rand(batch_size) * self.class_dim).type(torch.LongTensor)


                g_labels = self.one_hot_table[rand_labels].to(self.device)
                g_labels = g_labels.view(batch_size, self.class_dim, 1, 1)
                g_d_labels = self.d_input_table[rand_labels].to(self.device)


                g_result = self.generator(z, g_labels)
                d_result = self.discriminator(g_result, g_d_labels).squeeze()
                d_fake_loss = bce_loss(d_result, fake_y)

                d_loss = d_real_loss + d_fake_loss

                d_loss.backward()
                D_optimizer.step()
                # train the generator
                self.generator.zero_grad()

                z = torch.randn([batch_size, self.in_dim, 1, 1]).to(self.device)
                rand_labels = (torch.rand(batch_size) * 10).type(torch.LongTensor)

                g_labels = self.one_hot_table[rand_labels].to(self.device)
                g_labels = g_labels.view(batch_size, self.class_dim, 1, 1)
                g_d_labels = self.d_input_table[rand_labels].to(self.device)
                g_result = self.generator(z, g_labels)
                d_result = self.discriminator(g_result, g_d_labels).squeeze()
                g_loss = bce_loss(d_result, real_y)
                g_loss.backward()
                G_optimizer.step()

                print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                      %(epoch+1, num_epochs, i+1, len(data_loader), d_loss.item(), g_loss.item()))

    def reconstruct_loss(self, images, labels, lr=0.01, steps=1000):
        batch_size = images.size()[0]
        images = images.to(self.device)
        z = torch.zeros([batch_size, self.in_dim, 1, 1]).to(self.device)
        z.requires_grad = True
        g_labels = self.one_hot_table[labels].to(self.device)
        g_labels = g_labels.view(batch_size, self.class_dim, 1, 1)
        optimizer = optim.Adam([z], lr=lr)
        mse = nn.MSELoss()
        for step in range(steps):
            optimizer.zero_grad()
            g_result = self.generator(z, g_labels)
            loss = mse(g_result, images)
            loss.backward()
            optimizer.step()

            print('step: [%d/%d], loss: %.4f' % (step+1, steps, loss.item()))
        return g_result, loss.item()





















class Generator(nn.Module):
    def __init__(self, input_dim, class_dim, filter_nums, strides, kernel_size=4):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.filter_nums = filter_nums
        self.strides = strides
        self.kernel_size = kernel_size
        self.label_seq = nn.Sequential()
        self.input_seq = nn.Sequential()
        self.hidden_layer = nn.Sequential()
        for i in range(len(self.filter_nums)):
            padding=0 if self.strides[i]==1 else 1
            if i == 0:
                #for label
                self.label_seq.add_module('label_deconv',
                                          nn.ConvTranspose2d(self.class_dim,
                                                             self.filter_nums[i]//2,
                                                             self.kernel_size,
                                                             self.strides[i],
                                                             padding))
                self.label_seq.add_module('label_bn',
                                          nn.BatchNorm2d(self.filter_nums[i]//2, momentum=0.9))
                self.label_seq.add_module('label_act',
                                          nn.LeakyReLU(0.2))
                #for input
                self.input_seq.add_module('input_deconv',
                                          nn.ConvTranspose2d(self.input_dim,
                                                             self.filter_nums[i]//2,
                                                             self.kernel_size,
                                                             self.strides[i],
                                                             padding))
                self.input_seq.add_module('input_bn',
                                          nn.BatchNorm2d(self.filter_nums[i]//2, momentum=0.9))
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

    def forward(self, input, labels):
        # Concatenate label embedding and image to produce input
        x = self.input_seq(input)
        y = self.label_seq(labels)
        x = torch.cat([x, y], 1)
        x = self.hidden_layer(x)
        return (x + 1)/2



class Discriminator(nn.Module):
    def __init__(self, input_channel, class_dim, filter_nums, strides, kernel_size=4):
        super(Discriminator, self).__init__()
        self.input_channel = input_channel
        self.class_dim = class_dim
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
                self.label_seq.add_module('label_conv',
                                          nn.Conv2d(self.class_dim,
                                                    self.filter_nums[i]//2,
                                                    self.kernel_size,
                                                    self.strides[i],
                                                    padding))
                self.label_seq.add_module('label_bn',
                                          nn.BatchNorm2d(self.filter_nums[i]//2, momentum=0.9))
                self.label_seq.add_module('label_act',
                                          nn.LeakyReLU(0.2))
                # for input
                self.input_seq.add_module('input_conv',
                                          nn.Conv2d(self.input_channel,
                                                    self.filter_nums[i]//2,
                                                    self.kernel_size,
                                                    self.strides[i],
                                                    padding))
                self.input_seq.add_module('input_bn',
                                          nn.BatchNorm2d(self.filter_nums[i]//2, momentum=0.9))
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


    def forward(self, features, labels, extract_feature=False):
        x = self.input_seq(features)

        y = self.label_seq(labels)
        x = torch.cat([x, y], 1)
        feature = self.hidden_layer(x)
        out = self.output_layer(feature)
        out = torch.sigmoid(out)
        if extract_feature:
            return out, feature
        else:
            return out

