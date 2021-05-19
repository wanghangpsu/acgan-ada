'''
ACGAN implementation for feature based anomaly detection
based on paper: https://arxiv.org/abs/1610.09585
some code modified from https://github.com/clvrai/ACGAN-PyTorch
by Hang Wang   hzw81@psu.edu
'''



import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from CNN.resnet import BasicBlock
import torch.nn.functional as F

class ACGAN_Res:
    def __init__(self, in_dim, class_dim, CNN, feature_size=32, feature_id=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_dim = in_dim
        self.class_dim = class_dim


        self.generator = Generator(self.in_dim).to(self.device)
        self.discriminator = ResNet18().to(self.device)
        for p in self.generator.parameters():
            p.data.clamp_(-0.01, 0.01)
        #self.discriminator = ResNet18().to(self.device)
        self.CNN = CNN.to(self.device)
        self.feature_id = feature_id

        self.one_hot_table = torch.eye(self.class_dim, self.class_dim)
        self.feature_size = feature_size    #image size or feature map size
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

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



    def train(self, data_loader, lr=0.0002, num_epochs=40, test_loader=0):
        d_loss = []
        g_loss = []
        self.CNN.eval()
        self.discriminator.train()
        self.generator.train()

        dis_criterion = nn.BCELoss().to(self.device)
        aux_criterion = nn.NLLLoss().to(self.device)
        G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        for epoch in range(num_epochs):
            d_losses = []
            g_losses = []

            for i, (images, labels) in enumerate(data_loader):
                ###############################
                # train the discriminator
                ###############################
                self.discriminator.zero_grad()
                labels = labels.to(self.device)
                batch_size = images.size()[0]

                real_y = torch.ones(batch_size) + torch.randn(batch_size) * 0.1
                fake_y = torch.zeros(batch_size) + torch.randn(batch_size) * 0.1
                real_y = real_y.to(self.device)
                fake_y = fake_y.to(self.device)
                #extract features from the cnn
                #_, features = self.CNN(images, extract_feature=True)
                #features = features[self.feature_id]
                features = images.to(self.device)
                dis_out, aux_out = self.discriminator(features)
                dis_real_dloss = dis_criterion(dis_out, real_y)

                aux_real_dloss = aux_criterion(aux_out, labels)
                real_dloss = dis_real_dloss + 2 * aux_real_dloss
                real_dloss.backward()
                acc = compute_acc(aux_out, labels)

                #train D with fake samples
                # fake samples
                z = torch.randn([batch_size, self.in_dim]).to(self.device)
                rand_labels = (torch.rand(batch_size) * self.class_dim).type(torch.LongTensor)
                g_labels = self.one_hot_table[rand_labels].to(self.device)
                g_labels = g_labels.view(batch_size, self.class_dim)
                rand_labels = rand_labels.to(self.device)

                g_result = self.generator(z, g_labels)
                fake_dis, fake_aux = self.discriminator(g_result.detach())
                dis_fake_dloss = dis_criterion(fake_dis, fake_y)
                aux_fake_dloss = aux_criterion(fake_aux, rand_labels)

                #d_loss = dis_real_dloss + aux_real_dloss + dis_fake_dloss + aux_fake_dloss
                fake_dloss = dis_fake_dloss + 0.2 * aux_fake_dloss
                fake_dloss.backward()
                d_loss = fake_dloss + real_dloss
                D_optimizer.step()
                # train the generator

                self.generator.zero_grad()
                '''
                z = torch.randn([batch_size, self.in_dim]).to(self.device)
                rand_labels = (torch.rand(batch_size) * 10).type(torch.LongTensor)

                g_labels = self.one_hot_table[rand_labels].to(self.device)
                g_result = self.generator(z, g_labels)
                '''
                dis_output, aux_output = self.discriminator(g_result)
                dis_gloss = dis_criterion(dis_output, real_y)


                aux_gloss = aux_criterion(aux_output, rand_labels)
                g_loss = dis_gloss + 2*aux_gloss
                g_loss.backward()
                G_optimizer.step()

                print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f, acc: %.4f'
                      %(epoch+1, num_epochs, i+1, len(data_loader), d_loss.item(), g_loss.item(), acc))
            if test_loader!=0:
                self.acc(test_loader)
    def acc(self, test_loader):
        self.discriminator.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                _,outputs = self.discriminator(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(' Acc: %.3f%% (%d/%d)'
                      % (100. * correct / total, correct, total))
        self.discriminator.train()

    def reconstruct_loss(self, images, labels, lr=0.01, steps=1000):
        batch_size = images.size()[0]
        images = images.to(self.device)
        z = torch.zeros([batch_size, self.in_dim]).to(self.device)
        z.requires_grad = True
        g_labels = self.one_hot_table[labels].to(self.device)
        labels = labels.to(self.device)
        optimizer = optim.Adam([z], lr=lr)
        mse = nn.MSELoss()
        aux_loss = nn.NLLLoss()
        for step in range(steps):
            optimizer.zero_grad()
            g_result = self.generator(z, g_labels)
            loss = mse(g_result, images)
            loss.backward()
            optimizer.step()

            #print('step: [%d/%d], loss: %.4f' % (step+1, steps, loss.item()))
        dis, aux = self.discriminator(g_result)
        dis1, aux1 = self.discriminator(images)

        loss2 = aux1[0][labels.item()]
        print(loss.item(), loss2.item())
        return g_result, [loss.item(),  loss2.item(), dis1.item()]




#g_filters = [384, 192, 96, 48, 3]
#g_strids = [1, 2, 2, 2]


class GBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, first=False):
        super(GBasicBlock, self).__init__()

        if first:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=1, padding=0,
                                            bias=False)
        elif stride != 1:

            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=stride-1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=1, padding=1,
                                            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            if first:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, self.expansion * planes, kernel_size=4, stride=stride-1,
                                       padding=stride -2, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=4, stride=stride,
                                   padding=stride-1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, in_dim, block=GBasicBlock, num_blocks=(2, 2, 2, 2),  num_classes=10, img_channel=3):
        super(Generator, self).__init__()
        self.in_planes = 1024
        self.linear = nn.Linear(in_dim + num_classes, 1024)
        self.layer1 = self._make_layer(block, 512, num_blocks[0], stride=2, first=True)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.out_layer = nn.ConvTranspose2d(64, img_channel, 3, 1, 1)


    def _make_layer(self, block, planes, num_blocks, stride, first=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            if stride==2 and first:
                layers.append(block(self.in_planes, planes, stride, first=True))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inputs, labels):
        # Concatenate label embedding and image to produce input
        x = torch.cat([inputs, labels], 1)
        x = self.linear(x)
        x = x.view([-1, 1024, 1, 1])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.tanh(self.out_layer(x))

        #return x
        return (x + 1) / 2


# filters = [16, 32, 64, 128, 256, 512]
# strides = [2,1,2,1,2,1]



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4_1 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.in_planes = 256
        self.layer4_2 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear_re = nn.Linear(512*block.expansion, 1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out1 = self.layer4_1(out)
        out1 = F.avg_pool2d(out1, 4)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.layer4_2(out)
        out2 = F.avg_pool2d(out2, 4)
        out2 = out2.view(out2.size(0), -1)

        out_aux = self.linear(out1)
        out_dis = self.linear_re(out2)
        classes = self.softmax(out_aux)
        real = self.sigmoid(out_dis).view(-1, 1).squeeze(1)
        return real, classes

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


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



