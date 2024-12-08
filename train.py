import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

from model import UNet, FocalLoss, DiceLoss  # Import the model and custom losses

# ------------------------------
# Dataset and DataLoader
# ------------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])  # Example loading .npy files
        mask = np.load(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ------------------------------
# Training Pipeline
# ------------------------------
def train():
    # Paths to images and masks
    train_images = "/content/drive/MyDrive/DL_preprocessed/images"
    train_masks = "/content/drive/MyDrive/DL_preprocessed/masks"

    # Augmentation and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization
    ])

    train_dataset = CustomDataset(train_images, train_masks, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss, and optimizer
    model = UNet().to(device)
    focal_loss = FocalLoss(alpha=1, gamma=2)
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Training loop
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = focal_loss(outputs, masks) + dice_loss(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item()

        # Scheduler step
        scheduler.step(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # ------------------------------
    # Threshold Adjustment for Evaluation
    # ------------------------------
    def evaluate_model(model, data_loader, threshold=0.5):
        model.eval()
        f1_scores = []
        with torch.no_grad():
            for images, masks in data_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > threshold).float()

                # Calculate F1-score (simplified)
                tp = (preds * masks).sum().item()
                fp = ((1 - masks) * preds).sum().item()
                fn = (masks * (1 - preds)).sum().item()
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
                f1_scores.append(f1)

        print(f"Average F1-Score: {np.mean(f1_scores):.4f}")

    # Run evaluation
    evaluate_model(model, train_loader)

# ------------------------------
# Main Entry Point
# ------------------------------
if __name__ == "__main__":
    train()







# import os
# from pathlib import Path
# import torch
# import torch.optim as optim
# from torch import nn
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
# from sklearn.metrics import f1_score
# from unet import UNet

# # Paths
# image_folder = "/content/drive/MyDrive/DL_preprocessed/images"
# mask_folder = "/content/drive/MyDrive/DL_preprocessed/masks"
# model_folder = Path("model")
# model_folder.mkdir(exist_ok=True)
# model_path = "model/unet-voc.pt"
# best_model_path = "model/unet-best.pt"

# # Parameters
# saving_interval = 10
# epoch_number = 10
# batch_size = 2
# shuffle_data_loader = True
# learning_rate = 0.0001
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Create a weight tensor for the classes (background = 1, road = 10)
# weights = torch.tensor([1.0, 5.0]).to(device)  # Assign higher weight to the road class

# # Transformations
# transform_image = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB values
# ])

# transform_mask = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Grayscale()  # Convert to 1-channel (binary) mask
# ])

# # Custom Dataset for Images and Masks
# class CellDataset(Dataset):
#     def __init__(self, image_folder, mask_folder, transform_image, transform_mask):
#         self.image_folder = image_folder
#         self.mask_folder = mask_folder
#         self.image_files = sorted(os.listdir(image_folder))  # Sorted to match image and mask pairs
#         self.mask_files = sorted(os.listdir(mask_folder))  # Sorted to match image and mask pairs
#         self.transform_image = transform_image
#         self.transform_mask = transform_mask

#         # Ensure the filenames of images and masks match
#         for img, mask in zip(self.image_files, self.mask_files):
#             if img.split('.')[0].split('_')[-1] != mask.split('.')[0].split('_')[-1]:
#                 raise ValueError(f"Mismatch between image and mask filenames: {img}, {mask}")

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_folder, self.image_files[idx])
#         mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

#         image = Image.open(image_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")  # Grayscale mask for binary segmentation

#         if self.transform_image:
#             image = self.transform_image(image)
#         if self.transform_mask:
#             mask = self.transform_mask(mask)

#         return image, mask

# # F1 Score Calculation
# def calculate_f1_score(y_true, y_pred):
#     """
#     Calculate F1 Score for binary segmentation.
#     Args:
#         y_true (numpy array): Ground truth labels (H, W) or (N, H, W).
#         y_pred (numpy array): Predicted labels (H, W) or (N, H, W).
#     Returns:
#         dict: F1 score for the positive class and overall average.
#     """
#     y_true = y_true.flatten()
#     y_pred = y_pred.flatten()

#     if np.sum(y_pred == 1) == 0 or np.sum(y_true == 1) == 0:
#         return {"background": 1.0, "road": 0.0, "average": 0.5}  # Handle edge cases

#     f1_background = f1_score(y_true, y_pred, pos_label=0)
#     f1_road = f1_score(y_true, y_pred, pos_label=1)

#     return {
#         "background": f1_background,
#         "road": f1_road,
#         "average": (f1_background + f1_road) / 2,
#     }

# # Training Function
# def train():
#     dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)
#     cell_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

#     model = UNet(in_channels=3, out_channels=2)  # Binary segmentation
#     model.to(device)
#     if os.path.isfile(model_path):
#         model.load_state_dict(torch.load(model_path, map_location=device))

#     optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#     # Use the weighted CrossEntropyLoss
#     criterion = nn.CrossEntropyLoss(weight=weights)

#     best_f1 = 0.0
#     for epoch in range(epoch_number):
#         print(f"\nEpoch {epoch + 1}/{epoch_number}")
#         model.train()
#         epoch_losses = []
#         all_targets, all_predictions = [], []

#         for i, batch in enumerate(cell_loader):
#             inputs, targets = batch
#             inputs = inputs.to(device)
#             targets = targets.long().squeeze(dim=1).to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             epoch_losses.append(loss.item())

#             predictions = torch.argmax(outputs, dim=1)
#             all_targets.append(targets.cpu().numpy())
#             all_predictions.append(predictions.cpu().numpy())

#             if i % 10 == 0:
#                 print(f"Batch {i}: Loss = {loss.item():.4f}")

#         all_targets = np.concatenate(all_targets)
#         all_predictions = np.concatenate(all_predictions)

#         f1_scores = calculate_f1_score(all_targets, all_predictions)
#         print(f"F1 Scores: {f1_scores}")
#         avg_f1 = f1_scores["average"]

#         avg_loss = sum(epoch_losses) / len(epoch_losses)
#         print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

#         # Save the best model
#         if avg_f1 > best_f1:
#             best_f1 = avg_f1
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Best model saved with F1-score: {best_f1:.4f}")

#         if (epoch + 1) % saving_interval == 0:
#             torch.save(model.state_dict(), model_path)
#             print(f"Model saved at epoch {epoch + 1}")

#     print("Training complete.")
#     print(f"Best F1 Score: {best_f1:.4f}")
#     torch.save(model.state_dict(), model_path)
#     print("Final model saved.")

# Run the training process
# if __name__ == "__main__":
#     train()
