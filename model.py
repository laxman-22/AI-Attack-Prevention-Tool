import torchvision
from PIL import Image
from Adversarial_Example_Generation.image_attack import fgsm, pgd, cw, deep_fool
from torchvision.models.resnet import ResNet34_Weights
import torch

# Use the Pre-trained ResNet34 model
model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)

# Fine tune on tiny-imagenet (include data augmentation)


# Evaluate on fine-tuned cleaned imagenet dataset


# Train again on imagenet but this time generate adversarial examples (add augmented adversarial examples too)


# Evaluate on adversarial training



