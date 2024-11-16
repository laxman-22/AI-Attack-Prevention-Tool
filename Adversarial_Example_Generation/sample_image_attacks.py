import torchvision
from PIL import Image
from image_attack import fgsm, pgd, cw, deep_fool
from torchvision.models.resnet import ResNet34_Weights
import torch

model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)

model.eval()

img = Image.open('./tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG').convert('RGB')
img.show()

attacked = fgsm(model=model, image=img, label=torch.tensor([23]), epsilon=0.03)
attacked = Image.fromarray((attacked.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype("uint8"))
attacked.show()

attacked = pgd(model=model, image=img, label=torch.tensor([23]), epsilon=0.1, alpha=0.01, iterations=40)
attacked = Image.fromarray((attacked.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype("uint8"))
attacked.show()

attacked = cw(model=model, image=img, label=torch.tensor([23]), confidence=0, learning_rate=0.01, iterations=50)
attacked = Image.fromarray((attacked.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype("uint8"))
attacked.show()

attacked = deep_fool(model=model, image=img, label=torch.tensor([23]), overshoot=0.02, iterations=50)
attacked = Image.fromarray((attacked.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype("uint8"))
attacked.show()