import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from unet import UNet

# Paths and Parameters
test_image_folder = r"C:\Users\Lenovo\Desktop\unet\u-net\data\test_set_images"

model_path = "model/unet-voc.pt"
batch_size = 1
shuffle_data_loader = False

# Transformations: Match Training Preprocessing
transform_image = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512 (same as training)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize using training mean and std
])

transform_mask = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize masks (if ground truth is used)
    transforms.ToTensor()
])

# Custom Dataset for Testing
class TestDataset(Dataset):
    def __init__(self, root_folder, transform_image=None):
        self.root_folder = root_folder
        self.transform_image = transform_image

        # Collect all image paths from nested folders
        self.image_files = []
        for root, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):  # Support common image formats
                    self.image_files.append(os.path.join(root, file))

        assert len(self.image_files) > 0, f"No images found in {root_folder}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel RGB

        if self.transform_image:
            image = self.transform_image(image)

        return image, os.path.basename(image_path)  # Return image and filename for tracking


# Function to Apply a Colormap for Binary Segmentation
def apply_colormap(mask):
    """
    Convert binary segmentation mask to a grayscale or simple binary visualization.
    """
    mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255  # Convert mask to [0, 255]
    return Image.fromarray(mask)

# Prediction Function
def predict():
    # Load the Model
    model = UNet(in_channels=3, out_channels=2)  # Binary segmentation
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.eval()

    # DataLoader for Testing
    test_dataset = TestDataset(test_image_folder, transform_image=transform_image)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

    # Output Directory for Predicted Masks
    output_folder = r"C:\Users\Lenovo\Desktop\unet\u-net\data\test_set_predictions"
    os.makedirs(output_folder, exist_ok=True)

    for i, (input_image, filename) in enumerate(test_loader):
        input_image = input_image.to(torch.device("cpu"))  # Send to CPU (or GPU if available)
        with torch.no_grad():
            output = model(input_image)  # Forward pass
            predicted_mask = torch.argmax(output, dim=1)  # Get predicted class (binary)

        # Convert Predicted Mask to an Image
        predicted_colored_mask = apply_colormap(predicted_mask)

        # Save Predicted Mask
        predicted_colored_mask.save(os.path.join(output_folder, f"pred_{filename[0]}"))

        # Display Input Image and Prediction
        input_image_pil = transforms.ToPILImage()(input_image.squeeze().cpu())
        input_image_pil.show(title=f"Input Image: {filename[0]}")
        predicted_colored_mask.show(title=f"Predicted Mask: {filename[0]}")

        # Stop after a few predictions (optional)
        if i >= 10:
            break


if __name__ == "__main__":
    predict()
