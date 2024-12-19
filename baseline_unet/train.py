# Imports
import os
from pathlib import Path
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
from unet import UNet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

# Paths
# /content/drive/MyDrive/DL_data/DL_preprocessed_with_noise
# /content/drive/MyDrive/DL_data/processed_without_aug
# /content/drive/MyDrive/DL_data/DL_preprocessed_no_noise
image_folder = "/content/drive/MyDrive/DL_data/processed_without_aug/images"
mask_folder = "/content/drive/MyDrive/DL_data/processed_without_aug/masks"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet.pt"
best_model_path = "model/unet-best.pt"

# Parameters
saving_interval = 10
epoch_number = 10
batch_size = 8
shuffle_data_loader = True
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform_image = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

transform_mask = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Grayscale()  
])

# Custom Dataset for Images and Masks
class CellDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform_image, transform_mask):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))  # Sorted to match image and mask pairs
        self.mask_files = sorted(os.listdir(mask_folder))  # Sorted to match image and mask pairs
        self.transform_image = transform_image
        self.transform_mask = transform_mask

        for img, mask in zip(self.image_files, self.mask_files):
            if img.split('.')[0].split('')[-1] != mask.split('.')[0].split('')[-1]:
                raise ValueError(f"Mismatch between image and mask filenames: {img}, {mask}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

# F1 Score Calculation
def calculate_f1_score(y_true, y_pred):
    y_true = y_true.flatten() > 0.5
    y_pred = y_pred.flatten() > 0.5
    return {
        "background": f1_score(y_true, y_pred),
        "road": f1_score(y_true, y_pred),
        "average": f1_score(y_true, y_pred),
    }

# Visualization
loss_history = []
f1_history = []

def plot_metrics():
    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # F1 Score Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(f1_history) + 1), f1_history, marker='o', label='F1 Score', color='orange')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    
    # Save the figure
    plt.savefig('training_metrics.png')
    print("Plot saved as 'training_metrics.png'")
    plt.close()


# Training Function with Data Split
def train():
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    # Ensure images and masks match
    assert len(image_files) == len(mask_files), "Mismatch between images and masks."

    # splits (70% train, 15% val, 15% test)
    train_images, temp_images, train_masks, temp_masks = train_test_split(image_files, mask_files, test_size=0.3, random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(temp_images, temp_masks, test_size=0.5, random_state=42)

    # Create DataLoaders for each split
    train_dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)
    val_dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)
    test_dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)

    train_dataset.image_files = train_images
    train_dataset.mask_files = train_masks
    val_dataset.image_files = val_images
    val_dataset.mask_files = val_masks
    test_dataset.image_files = test_images
    test_dataset.mask_files = test_masks

    # Create DataLoader objects for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data_loader)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1)  # Binary segmentation
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCELoss()

    best_f1 = 0.0
    for epoch in range(epoch_number):
        print(f"\nEpoch {epoch + 1}/{epoch_number}")
        model.train()
        epoch_losses = []
        all_targets, all_predictions = [], []

        # Training loop
        for i, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            predictions = outputs > 0.5
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

            if i % 10 == 0:
                print(f"Batch {i}: Loss = {loss.item():.4f}")

        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        f1_scores = calculate_f1_score(all_targets, all_predictions)
        print(f"F1 Scores: {f1_scores}")
        avg_f1 = f1_scores["average"]

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        loss_history.append(avg_loss)
        f1_history.append(avg_f1)

        # Save the best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with F1-score: {best_f1:.4f}")

        if (epoch + 1) % saving_interval == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")
    
    print("Training complete.")
    print(f"Best F1 Score: {best_f1:.4f}")
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")

    # Plot the metrics
    plot_metrics()

    # Test Evaluation
    evaluate_on_test(model, test_loader)

def evaluate_on_test(model, test_loader):
    model.eval()
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            predictions = outputs > 0.5
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    f1_scores = calculate_f1_score(all_targets, all_predictions)
    print(f"Test F1 Scores: {f1_scores}")
    avg_f1 = f1_scores["average"]
    print(f"Test F1 Score: {avg_f1:.4f}")

# Run the training process
if __name__ == "__main__":
    train()