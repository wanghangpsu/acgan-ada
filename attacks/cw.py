import torch
import torch.nn as nn
import torch.optim as optim


class CW:
    def __init__(self, model, kappa=10, steps=1000, lr=0.01, targeted=1, c_id_=5):
        self.c_list = [1e-4, 2e-4, 5e-4,
                       1e-3, 2e-3, 5e-3,
                       1e-2, 2e-2, 5e-2,
                       1e-1, 2e-1, 5e-1,
                       1, 2, 5,
                       10, 20, 50]
        self.c_id = c_id_
        self.c = self.c_list[self.c_id]
        self.kappa = kappa
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.steps = steps
        self.targeted = targeted

    def attack(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)


        def f(x):
            outputs = self.model(x)

            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

            i = torch.max(torch.masked_select(outputs, (1 - one_hot_labels).bool()))
            j = torch.masked_select(outputs, one_hot_labels.bool())


            return torch.clamp(self.targeted * (i - j), min=-self.kappa)

        w = torch.zeros_like(images).to(self.device)
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=self.lr)
        c = self.c_list[self.c_id]
        c_id = self.c_id
        for step in range(self.steps):

            a = 1/2*(nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(c*f(a))


            cost =  loss1 + loss2
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            outputs = self.model(1/2*(nn.Tanh()(w) + 1))
            outputs = nn.Softmax(dim=1)(outputs)

            if step >= 200:
                if step % 50 == 0:
                    if outputs[0][labels[0].item()] < 0.9 and c_id < len(self.c_list)-1:
                        c_id += 1
                        c = self.c_list[c_id]
                if outputs[0][labels[0].item()] > 0.9:
                    break




        adv_images = (1/2*(nn.Tanh()(w) + 1)).detach()
        perturbation = adv_images - images


        return adv_images
