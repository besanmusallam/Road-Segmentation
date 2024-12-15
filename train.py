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
# from unet import UNet
from unet.unet_enhanced import UNet_enhanced as UNet
import matplotlib.pyplot as plt  # For visualization

# Paths
image_folder = "/content/drive/MyDrive/DL_data/DL_preprocessed_with_noise/images"
mask_folder = "/content/drive/MyDrive/DL_data/DL_preprocessed_with_noise/masks"
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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB values
])

transform_mask = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Grayscale()  # Convert to 1-channel (binary) mask
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
            if img.split('.')[0].split('_')[-1] != mask.split('.')[0].split('_')[-1]:
                raise ValueError(f"Mismatch between image and mask filenames: {img}, {mask}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask for binary segmentation

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


# Training Function
def train():
    dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)
    cell_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

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

        for i, batch in enumerate(cell_loader):
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

# Run the training process
if __name__ == "__main__":
    train()