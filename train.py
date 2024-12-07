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

# Paths
image_folder = "/content/drive/MyDrive/DL_preprocessed/images"
mask_folder = "/content/drive/MyDrive/DL_preprocessed/masks"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
best_model_path = "model/unet-best.pt"

# Parameters
saving_interval = 10
epoch_number = 10
batch_size = 2
shuffle_data_loader = True
learning_rate = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformations
transform_image = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB values
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
        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        self.transform_image = transform_image
        self.transform_mask = transform_mask

        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images and {len(self.mask_files)} masks."

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
    """
    Calculate F1 Score for binary segmentation.
    Args:
        y_true (numpy array): Ground truth labels (H, W) or (N, H, W).
        y_pred (numpy array): Predicted labels (H, W) or (N, H, W).
    Returns:
        dict: F1 score for the positive class and overall average.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    if np.sum(y_pred == 1) == 0 or np.sum(y_true == 1) == 0:
        return {"background": 1.0, "road": 0.0, "average": 0.5}  # Handle edge cases

    f1_background = f1_score(y_true, y_pred, pos_label=0)
    f1_road = f1_score(y_true, y_pred, pos_label=1)

    return {
        "background": f1_background,
        "road": f1_road,
        "average": (f1_background + f1_road) / 2,
    }

# Training Function
def train():
    dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)
    cell_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

    model = UNet(in_channels=3, out_channels=2)  # Binary segmentation
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(epoch_number):
        print(f"\nEpoch {epoch + 1}/{epoch_number}")
        model.train()
        epoch_losses = []
        all_targets, all_predictions = [], []

        for i, batch in enumerate(cell_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.long().squeeze(dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            predictions = torch.argmax(outputs, dim=1)
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

# Run the training process
if __name__ == "__main__":
    train()



# def calculate_f1_score(y_true, y_pred, num_classes):
#     """
#     Calculate F1 Score for semantic segmentation models.
    
#     Args:
#         y_true (numpy array): Ground truth labels (H, W) or (N, H, W).
#         y_pred (numpy array): Predicted labels (H, W) or (N, H, W).
#         num_classes (int): Number of classes in segmentation.

#     Returns:
#         dict: F1 scores for each class and their average.
#     """
#     if y_true.ndim == 3:
#         # Flatten all images for batch-wise comparison
#         y_true = y_true.reshape(-1)
#         y_pred = y_pred.reshape(-1)
#     elif y_true.ndim == 2:
#         y_true = y_true.flatten()
#         y_pred = y_pred.flatten()

#     f1_scores = {}
#     for i in range(num_classes):
#         f1_scores[f"class_{i}"] = f1_score(y_true == i, y_pred == i, average="binary")

#     f1_scores["average"] = np.mean(list(f1_scores.values()))
#     return f1_scores

# def train():
#     # Dataset and DataLoader
#     dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)
#     cell_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

#     # Model, Loss, Optimizer
#     model = UNet(in_channels=3, out_channels=2)  # Adjust output channels as needed
#     model.to(device)
#     if os.path.isfile(model_path):
#         model.load_state_dict(torch.load(model_path, map_location=device))

#     optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()

#     # Training Loop
#     for epoch in range(epoch_number):
#         print(f"Epoch {epoch + 1}/{epoch_number}")
#         model.train()
#         epoch_losses = []

#         # For calculating F1-score
#         all_targets = []
#         all_predictions = []

#         for i, batch in enumerate(cell_loader):
#             inputs, targets = batch
#             inputs = inputs.to(device)
#             targets = targets.long().squeeze(dim=1).to(device)  # Convert to integer labels

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             epoch_losses.append(loss.item())

#             # Store targets and predictions for F1 calculation
#             predictions = torch.argmax(outputs, dim=1)  # Get predicted class
#             all_targets.append(targets.cpu().numpy())
#             all_predictions.append(predictions.cpu().numpy())

#         # Flatten targets and predictions for F1 calculation
#         all_targets = np.concatenate(all_targets)
#         all_predictions = np.concatenate(all_predictions)

#         # Calculate F1 score
#         f1_scores = calculate_f1_score(all_targets, all_predictions, num_classes=2)
#         print(f"F1 Scores: {f1_scores}")

#         # Print average loss
#         avg_loss = sum(epoch_losses) / len(epoch_losses)
#         print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

#         # Save model at intervals
#         if (epoch + 1) % saving_interval == 0:
#             torch.save(model.state_dict(), model_path)
#             print(f"Model saved at epoch {epoch + 1}")

#     # Final Save
#     torch.save(model.state_dict(), model_path)
#     print("Final model saved.")

# Run the training process
if __name__ == "__main__":
    train()
