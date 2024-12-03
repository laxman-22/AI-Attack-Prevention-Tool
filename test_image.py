import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn

# Set device to GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
class FineTunedResNet50(nn.Module):
    def __init__(self):
        super(FineTunedResNet50, self).__init__()
        # Load ResNet50 with pretrained weights
        self.resnet = models.resnet50(weights='DEFAULT')

        # Freeze all layers except the final fully connected layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the fully connected layer for binary classification (clean vs attacked)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Binary output (attack vs clean)
        
        # Separate head for attack classification (5 types of attack)
        self.attack_output = nn.Linear(self.resnet.fc.in_features, 5)  # 5 attack types (including no_attack)

    def forward(self, x):
        # Pass input through ResNet up to the average pooling layer
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Average pool before the fully connected layer
        x = self.resnet.avgpool(x)
        
        # Flatten the features to pass to the fully connected layer
        x = torch.flatten(x, 1)  # Flatten the tensor, keeping the batch dimension
        
        # Binary classification (attack vs clean)
        binary_pred = torch.sigmoid(self.resnet.fc(x)).squeeze()  # Binary output
        
        # Attack type classification (for attacked images)
        attack_pred = self.attack_output(x)  # Attack classification (0-4 for attack types)
        
        return binary_pred, attack_pred

# Define the same image transformation as in training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image you want to test
img_path = './split_dataset/val/fgsm/n04070727_3578.JPEG'  # Replace with your image path
img = Image.open(img_path).convert("RGB")

# Apply the same transformations as during training
img = transform(img).unsqueeze(0)  # Add batch dimension

# Load the trained model
model = FineTunedResNet50().to(device)
model.load_state_dict(torch.load('best_binary_model.pth', weights_only=True))  # Load the best model

# Move image to GPU if available
img = img.to(device)

# Set model to evaluation mode
model.eval()

# Make prediction
with torch.no_grad():
    binary_pred, attack_pred = model(img)

# Convert predictions
binary_pred = binary_pred.item()  # Convert tensor to scalar
attack_pred = attack_pred.argmax(dim=1).item()  # Get the index of the max value for attack type

# Output results
print(f"Binary prediction (0 = clean, 1 = attacked): {binary_pred}")
print(f"Attack prediction (0 = no_attack, 1 = deepfool, 2 = fgsm, 3 = pgd, 4 = cw): {attack_pred}")
