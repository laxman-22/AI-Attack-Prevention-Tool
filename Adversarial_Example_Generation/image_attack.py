from torchvision.transforms import transforms
import torchattacks

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def fgsm(model, image, epsilon, label):
    img = transform(image).unsqueeze(0)
    attack = torchattacks.FGSM(model, eps=epsilon)
    attacked_img = attack(img, label)
    return attacked_img

def pgd(model, image, epsilon, alpha, iterations, label):
    img = transform(image).unsqueeze(0)
    attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=iterations)
    attacked_img = attack(img, label)
    return attacked_img

def cw(model, image, label, confidence, learning_rate, iterations):
    img = transform(image).unsqueeze(0)
    attack = torchattacks.CW(model, kappa=confidence, lr=learning_rate, steps=iterations)
    attacked_img = attack(img, label)
    return attacked_img

def deep_fool(model, image, label, overshoot, iterations):
    img = transform(image).unsqueeze(0)
    attack = torchattacks.DeepFool(model, overshoot=overshoot, steps=iterations)
    attacked_img = attack(img, label)
    return attacked_img