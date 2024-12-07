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
    def __init__(self, image_folder, mask_folder=None, transform_image=None, transform_mask=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder)) if mask_folder else None
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel RGB

        if self.transform_image:
            image = self.transform_image(image)

        if self.mask_folder:
            mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
            mask = Image.open(mask_path).convert("L")  # Grayscale for binary segmentation
            if self.transform_mask:
                mask = self.transform_mask(mask)
            return image, mask

        return image

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

    for i, input_image in enumerate(test_loader):
        input_image = input_image.to(torch.device("cpu"))  # Send to CPU (or GPU if available)
        with torch.no_grad():
            output = model(input_image)  # Forward pass
            predicted_mask = torch.argmax(output, dim=1)  # Get predicted class (binary)

        # Visualize Input Image
        input_image_pil = transforms.ToPILImage()(input_image.squeeze().cpu())
        input_image_pil.show(title="Input Image")

        # Visualize Predicted Mask
        predicted_colored_mask = apply_colormap(predicted_mask)
        predicted_colored_mask.show(title="Predicted Mask")

        # Stop after 10 predictions (optional)
        if i >= 10:
            break

if __name__ == "__main__":
    predict()



# import numpy as np
# from PIL import Image
# import torch
# from torchvision import transforms, datasets
# from matplotlib import pyplot as plt
# from unet import UNet

# # Paths and Parameters
# data_folder = r"C:\Users\Lenovo\Desktop\unet\u-net\data\test_set_images"
# model_path = "model/unet-voc.pt"
# batch_size = 1
# shuffle_data_loader = False

# # Transformations
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
# ])

# # Dataset (VOC 2007)
# dataset = datasets.VOCSegmentation(
#     root=data_folder,
#     year="2007",
#     download=True,
#     image_set="val",  # Using 'val' for validation/testing
#     transform=transform,
#     target_transform=transform,  # Leave masks unchanged
# )

# # Function to Apply a Color Map for Multi-class Segmentation
# def apply_colormap(mask, num_classes=21):
#     """Apply a colormap to the segmentation mask."""
#     colormap = plt.cm.get_cmap("tab20", num_classes)  # Use 'tab20' colormap
#     mask = mask.squeeze().cpu().numpy().astype(int)  # Convert to numpy
#     colored_mask = colormap(mask / num_classes)[:, :, :3]  # Normalize and apply colormap
#     colored_mask = (colored_mask * 255).astype(np.uint8)  # Scale to 0-255
#     return Image.fromarray(colored_mask)

# # Prediction Function
# def predict():
#     # Load the Model
#     model = UNet(in_channels=3, out_channels=21)  # 21 classes (VOC)
#     checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
#     model.load_state_dict(checkpoint)
#     model.eval()

#     # DataLoader
#     cell_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

#     for i, (input_image, target_mask) in enumerate(cell_loader):
#         input_image = input_image  # Input tensor
#         with torch.no_grad():
#             output = model(input_image)  # Forward pass
#             predicted_mask = output.argmax(dim=1)  # Get class predictions

#         # Visualize Input Image
#         input_image = transforms.ToPILImage()(input_image.squeeze())
#         input_image.show(title="Input Image")

#         # Visualize Predicted Mask
#         predicted_colored_mask = apply_colormap(predicted_mask, num_classes=21)
#         predicted_colored_mask.show(title="Predicted Mask")

#         # Visualize Ground Truth (Optional)
#         target_colored_mask = apply_colormap(target_mask.argmax(dim=0), num_classes=21)
#         target_colored_mask.show(title="Ground Truth Mask")

#         # Stop after 10 predictions (Optional)
#         if i >= 10:
#             break

# if __name__ == "__main__":
#     predict()
