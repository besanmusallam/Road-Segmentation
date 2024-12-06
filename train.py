import os
from pathlib import Path
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from unet import UNet

# Paths
# data_folder = r"C:\Users\Lenovo\Desktop\unet\u-net\data"
# image_folder = os.path.join(data_folder, "images")
# mask_folder = os.path.join(data_folder, "masks")
image_folder = "/content/drive/MyDrive/DL_preprocessed/images"
mask_folder = "/content/drive/MyDrive/DL_preprocessed/masks"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"

# Parameters
saving_interval = 10
epoch_number = 10
batch_size = 32
shuffle_data_loader = True
learning_rate = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformations
transform_image = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

# Training Function
def train():
    # Dataset and DataLoader
    dataset = CellDataset(image_folder, mask_folder, transform_image, transform_mask)
    cell_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

    # Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=2)  # Adjust output channels as needed
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(epoch_number):
        print(f"Epoch {epoch + 1}/{epoch_number}")
        model.train()
        epoch_losses = []

        for i, batch in enumerate(cell_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.long().squeeze(dim=1).to(device)  # Convert to integer labels

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Print average loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        # Save model at intervals
        if (epoch + 1) % saving_interval == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")

    # Final Save
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")

# Run the training process
if __name__ == "__main__":
    train()
