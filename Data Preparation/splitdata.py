import os
import shutil
import random

def split_dataset(source_dir, dest_dir, val_split=0.2):
    """
    Splits a dataset into train and validation sets.

    :param source_dir: Path to the dataset with class subdirectories.
    :param dest_dir: Path where the split dataset will be saved.
    :param val_split: Proportion of data to use for validation (default: 0.2).
    """
    random.seed(42)  # Ensure reproducibility

    # Create train and val directories
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Loop through each class folder
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Create class subdirectories in train and val
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get all files in the class directory
        files = os.listdir(class_path)
        random.shuffle(files)

        # Split files into train and val
        split_idx = int(len(files) * (1 - val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # Move files to their respective directories
        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(train_class_dir, file))
        for file in val_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(val_class_dir, file))

        print(f"Processed class '{class_name}': {len(train_files)} train, {len(val_files)} val")

    print("Dataset split completed!")

# Example usage
source_directory = "./dataset"  # Original dataset (e.g., "dataset/")
destination_directory = "./split_dataset"  # New dataset structure (e.g., "split_dataset/")
split_dataset(source_directory, destination_directory)