import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
from unet import UNet

# Paths and Parameters
data_folder = "/content/test_set_images.zip"
model_path = "model/unet-voc.pt"
batch_size = 1
shuffle_data_loader = False

# Transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Dataset (VOC 2007)
dataset = datasets.VOCSegmentation(
    root=data_folder,
    year="2007",
    download=True,
    image_set="val",  # Using 'val' for validation/testing
    transform=transform,
    target_transform=transform,  # Leave masks unchanged
)

# Function to Apply a Color Map for Multi-class Segmentation
def apply_colormap(mask, num_classes=21):
    """Apply a colormap to the segmentation mask."""
    colormap = plt.cm.get_cmap("tab20", num_classes)  # Use 'tab20' colormap
    mask = mask.squeeze().cpu().numpy().astype(int)  # Convert to numpy
    colored_mask = colormap(mask / num_classes)[:, :, :3]  # Normalize and apply colormap
    colored_mask = (colored_mask * 255).astype(np.uint8)  # Scale to 0-255
    return Image.fromarray(colored_mask)

# Prediction Function
def predict():
    # Load the Model
    model = UNet(in_channels=3, out_channels=21)  # 21 classes (VOC)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    model.eval()

    # DataLoader
    cell_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data_loader)

    for i, (input_image, target_mask) in enumerate(cell_loader):
        input_image = input_image  # Input tensor
        with torch.no_grad():
            output = model(input_image)  # Forward pass
            predicted_mask = output.argmax(dim=1)  # Get class predictions

        # Visualize Input Image
        input_image = transforms.ToPILImage()(input_image.squeeze())
        input_image.show(title="Input Image")

        # Visualize Predicted Mask
        predicted_colored_mask = apply_colormap(predicted_mask, num_classes=21)
        predicted_colored_mask.show(title="Predicted Mask")

        # Visualize Ground Truth (Optional)
        target_colored_mask = apply_colormap(target_mask.argmax(dim=0), num_classes=21)
        target_colored_mask.show(title="Ground Truth Mask")

        # Stop after 10 predictions (Optional)
        if i >= 10:
            break

if __name__ == "__main__":
    predict()
