import torch
import torchvision
from PIL import Image
import os
import sys
import random
import threading
import subprocess
from collections import defaultdict
from Adversarial_Example_Generation.image_attack import deep_fool, preprocess_image
from torchvision.models.resnet import ResNet34_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_image_count = 0


# Function to monitor GPU utilization
def monitor_gpu_memory_utilization():
    while True:
        try:
            # Use nvidia-smi to get the memory utilization
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            memory_used = int(result.stdout.strip().split("\n")[0])  # Get the memory usage value in MB

            # Use nvidia-smi to get the total memory
            total_memory_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            total_memory = int(total_memory_result.stdout.strip().split("\n")[0])  # Total memory in MB

            # Calculate percentage of memory used
            memory_utilization = (memory_used / total_memory) * 100

            if memory_utilization > 95:
                print(f"GPU memory utilization is {memory_utilization:.2f}%. Restarting script...")
                os.execv(sys.executable, ['python'] + sys.argv)  # Restart the script
        except Exception as e:
            print(f"Error monitoring GPU memory utilization: {e}")
        torch.cuda.empty_cache()  # Free up any unused GPU memory

# Function to get the predicted label
def get_predicted_label(model, image):
    image_tensor = preprocess_image(image).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_label = output.max(1)  # Get the class with the highest probability
    return predicted_label.item()


# Apply the DeepFool attack to selected images
def apply_deep_fool_to_directory(input_dir, output_dir, model, overshoot=0.02, iterations=30):
    global processed_image_count
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    processed_classes = defaultdict(int)
    for filename in os.listdir(output_dir):
        if filename.endswith('JPEG'):
            class_id = filename.split('_')[0]
            processed_classes[class_id] += 1

    class_files = defaultdict(list)
    for filename in os.listdir(input_dir):
        if filename.endswith('JPEG'):
            class_id = filename.split('_')[0]
            class_files[class_id].append(filename)

    for class_id, files in class_files.items():
        if processed_classes[class_id] >= 5:
            print(f"Skipping class {class_id}, already has 5 processed files.")
            continue

        random.shuffle(files)
        for filename in files:

            if processed_classes[class_id] >= 5:
                break

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing file: {filename} from class {class_id}")

            try:
                image = Image.open(input_path).convert("RGB")
                label = get_predicted_label(model, image)
                print(f"Predicted label for {filename}: {label}")

                image_tensor = preprocess_image(image).to(device)
                image_tensor.requires_grad = True

                # attacked = cw(model=model, images=image_tensor, label=torch.tensor([label]), confidence=random.randint(0, 10), learning_rate=random.uniform(0.001, 0.01), iterations=random.randint(10, 1000))
                attacked = deep_fool(model, image_tensor, torch.tensor([label], device=device), overshoot, iterations)

                attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                attacked_img = (attacked_img * 255).astype("uint8")
                attacked_pil = Image.fromarray(attacked_img)

                attacked_pil.save(output_path)
                print(f"Processed and saved: {filename}")

                del image_tensor, attacked, attacked_img, attacked_pil
                torch.cuda.empty_cache()

                processed_classes[class_id] += 1
                processed_image_count += 1

            except Exception as e:
                print(f"Error occurred for {filename}: {e}")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    # Start GPU monitoring in a separate thread
    gpu_monitor_thread = threading.Thread(target=monitor_gpu_memory_utilization, daemon=True)
    gpu_monitor_thread.start()

    # Load the pre-trained ResNet-34 model
    model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.eval()
    model = model.to(device)

    input_directory = "./clean"
    output_directory = "./dataset/deepfool"
    try:
        apply_deep_fool_to_directory(input_directory, output_directory, model)
    except SystemExit:
        torch.cuda.empty_cache()
        os.execv(sys.executable, ['python'] + sys.argv)