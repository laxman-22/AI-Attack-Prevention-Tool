import torchattacks
from torchvision import transforms

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def fgsm(model, images, epsilon, label):
    attack = torchattacks.FGSM(model, eps=epsilon)
    attacked_img = attack(images, label)
    return attacked_img

def pgd(model, images, epsilon, alpha, iterations, label):
    attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=iterations)
    attacked_img = attack(images, label)
    return attacked_img

def cw(model, images, label, confidence, learning_rate, iterations):
    attack = torchattacks.CW(model, kappa=confidence, lr=learning_rate, steps=iterations)
    attacked_img = attack(images, label)
    return attacked_img

def deep_fool(model, images, label, overshoot, iterations):
    attack = torchattacks.DeepFool(model, overshoot=overshoot, steps=iterations)
    attacked_img = attack(images, label)
    return attacked_img