import os
import random
import shutil

# Set your source directory where the class folders are located
source_dir = './ILSVRC2017_DET/ILSVRC/Data/DET/train/ILSVRC2013_train'
# Set the destination directory where you want to copy the images to
destination_dir = './dataset/clean'

# Loop through each class directory
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    # Only process directories
    if os.path.isdir(class_path):
        # Get all images in the class directory
        images = [f for f in os.listdir(class_path) if f.endswith("JPEG")]
        
        # Randomly sample 100 images
        selected_images = random.sample(images, min(500, len(images)))  # Ensure not to sample more than available
        
        # Copy the selected images to the destination directory
        for image in selected_images:
            image_path = os.path.join(class_path, image)
            shutil.copy(image_path, os.path.join(destination_dir, image))

        print(f"Copied {len(selected_images)} images from {class_name}")

print("Image copying completed.")