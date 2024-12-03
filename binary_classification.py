import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score

# Set device to GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Small rotation
    transforms.GaussianBlur(3),  # Simulate blurriness
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class AttackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir))  # ['no_attack', 'deepfool', 'fgsm', ...]

        # Define a mapping for attack types
        self.attack_types = {'no_attack': 0, 'deepfool': 1, 'fgsm': 2, 'pgd': 3, 'cw': 4}

        self.image_paths = []
        self.labels = []

        for attack_name in self.class_names:
            attack_folder = os.path.join(root_dir, attack_name)
            for img_name in os.listdir(attack_folder):
                img_path = os.path.join(attack_folder, img_name)
                self.image_paths.append(img_path)

                # Binary label: 0 for clean (no_attack), 1 for attacked
                binary_label = 0 if attack_name == 'no_attack' else 1
                attack_label = self.attack_types.get(attack_name, -1)
                self.labels.append((binary_label, attack_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        binary_label, attack_label = self.labels[idx]
        
        return img, {'binary': torch.tensor(binary_label), 'attack_type': torch.tensor(attack_label)}

train_data = AttackDataset(root_dir='./split_dataset/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = AttackDataset(root_dir='./split_dataset/val', transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

class FineTunedResNet50(nn.Module):
    def __init__(self):
        super(FineTunedResNet50, self).__init__()
        
        # Load ResNet50 with pretrained weights
        self.resnet = models.resnet50(weights='DEFAULT')
        
        # Freeze all layers except the final fully connected layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer2.parameters():
            param.requires_grad = True
        
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
    
def custom_loss(binary_pred, attack_pred, labels):
    binary_loss = F.binary_cross_entropy(binary_pred, labels['binary'].float())
    attack_loss = F.cross_entropy(attack_pred, labels['attack_type'], reduction='mean')
    attack_weight = 2.0
    return binary_loss + attack_weight * attack_loss

def evaluate_model(model, val_loader):
    model.eval()  # Set the model to evaluation mode
    binary_preds = []
    attack_preds = []
    binary_labels = []
    attack_labels = []

    with torch.no_grad():
        for data, labels in val_loader:
            imgs, targets = data, labels
            imgs, targets = imgs.to(device), {key: val.to(device) for key, val in targets.items()}  # Move to GPU
            
            binary_pred, attack_pred = model(imgs)
            
            # Collect predictions and ground truths
            binary_preds.append(binary_pred.cpu().numpy())
            attack_preds.append(attack_pred.cpu().numpy())
            binary_labels.append(targets['binary'].cpu().numpy())
            attack_labels.append(targets['attack_type'].cpu().numpy())

    # Flatten the lists
    binary_preds = np.concatenate(binary_preds)
    attack_preds = np.concatenate(attack_preds)
    binary_labels = np.concatenate(binary_labels)
    attack_labels = np.concatenate(attack_labels)

    # Compute accuracy for binary classification
    binary_accuracy = accuracy_score(binary_labels, (binary_preds > 0.5).astype(int))

    # Compute accuracy for attack type classification (only for attacked images)
    attacked = binary_labels == 1
    attack_preds = np.argmax(attack_preds, axis=1)
    attack_accuracy = accuracy_score(attack_labels[attacked], attack_preds[attacked])

    return binary_accuracy, attack_accuracy

model = FineTunedResNet50().to(device)  # Move the model to GPU
model.load_state_dict(torch.load('imp_binary_model.pth', weights_only=True))
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 10
best_binary_accuracy = 0.9652

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0
    binary_train_preds = []
    attack_train_preds = []
    binary_train_labels = []
    attack_train_labels = []

    for data, labels in train_loader:
        imgs, targets = data, labels
        imgs, targets = imgs.to(device), {key: val.to(device) for key, val in targets.items()}  # Move to GPU
        optimizer.zero_grad()
        
        # Forward pass
        binary_pred, attack_pred = model(imgs)
        
        binary_train_preds.append(binary_pred.cpu().detach().numpy())
        attack_train_preds.append(attack_pred.cpu().detach().numpy())
        binary_train_labels.append(targets['binary'].cpu().numpy())
        attack_train_labels.append(targets['attack_type'].cpu().numpy())
        # Calculate loss
        loss = custom_loss(binary_pred, attack_pred, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate the average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    
    binary_train_preds = np.concatenate(binary_train_preds)
    attack_train_preds = np.concatenate(attack_train_preds)
    binary_train_labels = np.concatenate(binary_train_labels)
    attack_train_labels = np.concatenate(attack_train_labels)

    # Compute training accuracies
    train_binary_accuracy = accuracy_score(binary_train_labels, (binary_train_preds > 0.5).astype(int))
    train_attacked = binary_train_labels == 1
    train_attack_preds = np.argmax(attack_train_preds, axis=1)
    train_attack_accuracy = accuracy_score(attack_train_labels[train_attacked], train_attack_preds[train_attacked])

    # Evaluate the model on the validation set
    val_binary_accuracy, val_attack_accuracy = evaluate_model(model, val_loader)
    

    # Print the results for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, "
          f"Train Binary Accuracy: {train_binary_accuracy:.4f}, Train Attack Accuracy: {train_attack_accuracy:.4f}, "
          f"Val Binary Accuracy: {val_binary_accuracy:.4f}, Val Attack Accuracy: {val_attack_accuracy:.4f}")
    
    # If binary accuracy improves, save the model
    if val_binary_accuracy > best_binary_accuracy:
        best_binary_accuracy = val_binary_accuracy
        # Save the best model
        torch.save(model.state_dict(), 'imp_binary_model.pth')
        print(f"New best binary accuracy! Model saved as 'imp_binary_model.pth'")
