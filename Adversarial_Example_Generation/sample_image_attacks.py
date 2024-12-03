import torchvision
from PIL import Image
from image_attack import fgsm, pgd, cw, deep_fool, preprocess_image
from torchvision.models.resnet import ResNet34_Weights
import torch

model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)

model.eval()

img = Image.open('./dataset/clean/n00007846_38367.JPEG').convert('RGB')
img.show()
img_tensor = preprocess_image(img)

attacked = fgsm(model=model, images=img_tensor, label=torch.tensor([23]), epsilon=0.03)
attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
attacked_img = (attacked_img * 255).astype("uint8")
attacked_pil = Image.fromarray(attacked_img)
attacked_pil.show()

attacked = pgd(model=model, images=img_tensor, label=torch.tensor([23]), epsilon=0.1, alpha=0.01, iterations=40)
attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
attacked_img = (attacked_img * 255).astype("uint8")
attacked_pil = Image.fromarray(attacked_img)
attacked_pil.show()

attacked = cw(model=model, images=img_tensor, label=torch.tensor([23]), confidence=0, learning_rate=0.01, iterations=50)
attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
attacked_img = (attacked_img * 255).astype("uint8")
attacked_pil = Image.fromarray(attacked_img)
attacked_pil.show()

attacked = deep_fool(model=model, images=img_tensor, label=torch.tensor([23]), overshoot=0.02, iterations=50)
attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
attacked_img = (attacked_img * 255).astype("uint8")
attacked_pil = Image.fromarray(attacked_img)
attacked_pil.show()