import torchvision
from PIL import Image
from Adversarial_Example_Generation.image_attack import fgsm, pgd, cw, deep_fool
from torchvision.models.resnet import ResNet34_Weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# Use the Pre-trained ResNet34 model
model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)


for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer (fully connected layer)
for param in model.fc.parameters():
    param.requires_grad = True

# Fine tune on tiny-imagenet (include data augmentation)
model.fc = nn.Linear(model.fc.in_features, 200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD([
    {'params': model.fc.parameters(), 'lr': 1e-2}
], lr=1e-4, momentum=0.9)

epochs = 10

writer = SummaryWriter()

# Evaluate on fine-tuned cleaned imagenet dataset


# Train again on imagenet but this time generate adversarial examples (add augmented adversarial examples too)


# Evaluate on adversarial training



