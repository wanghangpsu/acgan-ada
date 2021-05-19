import torch
import torch.nn as nn
import torch.optim as optim


class FGSM:
    def __init__(self, model, epsilon=0.3, targeted=1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.loss = torch.nn.NLLLoss()
    def attack(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.requires_grad = True

        output = self.model(images)
        loss = -1 * self.targeted * self.loss(output, labels)
        self.model.zero_grad()
        loss.backward()
        images_grad = images.grad.data

        sign_data_grad = images_grad.sign()

        adv_images = images + self.epsilon * sign_data_grad
        adv_images = torch.clamp(adv_images, 0, 1)
        return adv_images.detach()
