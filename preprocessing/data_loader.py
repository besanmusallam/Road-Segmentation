import os
import cv2
import math
import numpy as np
import random
# import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

# Function to load data
def load_data(image_dir, mask_dir, img_size=(400, 400)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    assert len(image_files) == len(mask_files), "Number of images and masks do not match!"
    for img, mask in zip(image_files, mask_files):
        assert img.split('.')[0].split('_')[-1] == mask.split('.')[0].split('_')[-1], f"Mismatch: {img} and {mask}"

    images = [cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_GRAYSCALE) for f in image_files]
    masks = [cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in mask_files]

    # Add channel dimension
    images = [img[..., np.newaxis] for img in images]
    masks = [mask[..., np.newaxis] for mask in masks]

    return np.array(images), np.array(masks)

# Custom ImageDataGenerator class without Data Augmentation
class ImageDataGenerator(Sequence):
    def __init__(self, images, masks, target_size=(512, 512), batch_size=32, 
                 augment=False, save_dir=None, preprocessed_save_dir=None, augmentation_count=4):
        self.images = images
        self.masks = masks
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.save_dir = save_dir
        self.preprocessed_save_dir = preprocessed_save_dir
        self.augmentation_count = 0  # To keep track of augmented images
        self.augment_count = augmentation_count  # Number of augmentations per image

        if self.save_dir:
            # Ensure save directory exists
            os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "masks"), exist_ok=True)

        if self.preprocessed_save_dir:
            os.makedirs(os.path.join(self.preprocessed_save_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.preprocessed_save_dir, "masks"), exist_ok=True)

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.images))
        batch_images = self.images[start:end]
        batch_masks = self.masks[start:end]
        return self._preprocess_batch(batch_images, batch_masks)

    def _preprocess_batch(self, batch_images, batch_masks):
        processed_images = []
        processed_masks = []

        for img, mask in zip(batch_images, batch_masks):
            resized_img = cv2.resize(img, self.target_size)
            resized_mask = cv2.resize(mask, self.target_size)

            # Save preprocessed data (non-augmented)
            if self.preprocessed_save_dir:
                self._save_preprocessed_data(resized_img, resized_mask)

            if self.augment:
                # Apply augmentations
                for i in range(self.augment_count):
                    augmented_img, augmented_mask = self._augment_and_save(
                        resized_img, resized_mask, self.save_dir, self.augmentation_count
                    )
                    self.augmentation_count += 1
                    processed_images.append(augmented_img / 255.0)
                    processed_masks.append(augmented_mask / 255.0)
            else:
                processed_images.append(resized_img / 255.0)
                processed_masks.append(resized_mask / 255.0)

        return np.array(processed_images), np.array(processed_masks)
    
    def _save_preprocessed_data(self, image, mask):
        """Save the preprocessed images and masks in a different directory."""
        if self.preprocessed_save_dir:
            # Use a unique count for saving preprocessed data
            image_filename = os.path.join(self.preprocessed_save_dir, "images", f"image_{self.augmentation_count}.png")
            mask_filename = os.path.join(self.preprocessed_save_dir, "masks", f"mask_{self.augmentation_count}.png")
            cv2.imwrite(image_filename, image)
            cv2.imwrite(mask_filename, mask)
    
    def _augment_and_save(self, image, mask, save_dir, count):
        # Flip augmentation
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Rotation augmentation
        angle = random.randint(-30, 30)
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        mask = cv2.warpAffine(mask, rotation_matrix, (cols, rows))

        # Translation augmentation
        max_trans_x = int(0.2 * cols)
        max_trans_y = int(0.2 * rows)
        trans_x = random.randint(-max_trans_x, max_trans_x)
        trans_y = random.randint(-max_trans_y, max_trans_y)
        translation_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        mask = cv2.warpAffine(mask, translation_matrix, (cols, rows))

        # Brightness augmentation
        if random.random() > 0.5:
            factor = 0.5 + random.random()  # Random factor between 0.5 and 1.5
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Remove the Gaussian noise augmentation (this part is deleted)

        # Save augmented images and masks
        augmented_image_filename = os.path.join(save_dir, "images", f"image_aug_{count}.png")
        augmented_mask_filename = os.path.join(save_dir, "masks", f"mask_aug_{count}.png")
        cv2.imwrite(augmented_image_filename, image)
        cv2.imwrite(augmented_mask_filename, mask)

        return image, mask


# import os
# import cv2
# import math
# import numpy as np
# import random
# import imgaug.augmenters as iaa
# import matplotlib.pyplot as plt
# from tensorflow.keras.utils import Sequence

# # Function to load data
# def load_data(image_dir, mask_dir, img_size=(400, 400)):
#     image_files = sorted(os.listdir(image_dir))
#     mask_files = sorted(os.listdir(mask_dir))

#     assert len(image_files) == len(mask_files), "Number of images and masks do not match!"
#     for img, mask in zip(image_files, mask_files):
#         assert img.split('.')[0].split('_')[-1] == mask.split('.')[0].split('_')[-1], f"Mismatch: {img} and {mask}"

#     images = [cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_GRAYSCALE) for f in image_files]
#     masks = [cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE) for f in mask_files]

#     # Add channel dimension
#     images = [img[..., np.newaxis] for img in images]
#     masks = [mask[..., np.newaxis] for mask in masks]

#     return np.array(images), np.array(masks)

# # Custom ImageDataGenerator class without Data Augmentation
# class ImageDataGenerator(Sequence):
#     def __init__(self, images, masks, target_size=(512, 512), batch_size=32, 
#                  augment=False, save_dir=None, preprocessed_save_dir=None, augmentation_count=4):
#         self.images = images
#         self.masks = masks
#         self.target_size = target_size
#         self.batch_size = batch_size
#         self.augment = augment
#         self.save_dir = save_dir
#         self.preprocessed_save_dir = preprocessed_save_dir
#         self.augmentation_count = 0  # To keep track of augmented images
#         self.augment_count = augmentation_count  # Number of augmentations per image

#         if self.save_dir:
#             # Ensure save directory exists
#             os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
#             os.makedirs(os.path.join(self.save_dir, "masks"), exist_ok=True)

#         if self.preprocessed_save_dir:
#             os.makedirs(os.path.join(self.preprocessed_save_dir, "images"), exist_ok=True)
#             os.makedirs(os.path.join(self.preprocessed_save_dir, "masks"), exist_ok=True)

#     def __len__(self):
#         return math.ceil(len(self.images) / self.batch_size)

#     def __getitem__(self, idx):
#         start = idx * self.batch_size
#         end = min((idx + 1) * self.batch_size, len(self.images))
#         batch_images = self.images[start:end]
#         batch_masks = self.masks[start:end]
#         return self._preprocess_batch(batch_images, batch_masks)

#     def _preprocess_batch(self, batch_images, batch_masks):
#         processed_images = []
#         processed_masks = []

#         for img, mask in zip(batch_images, batch_masks):
#             resized_img = cv2.resize(img, self.target_size)
#             resized_mask = cv2.resize(mask, self.target_size)

#             # Save preprocessed data (non-augmented)
#             if self.preprocessed_save_dir:
#                 self._save_preprocessed_data(resized_img, resized_mask)

#             if self.augment:
#                 # Apply augmentations
#                 for i in range(self.augment_count):
#                     augmented_img, augmented_mask = self._augment_and_save(
#                         resized_img, resized_mask, self.save_dir, self.augmentation_count
#                     )
#                     self.augmentation_count += 1
#                     processed_images.append(augmented_img / 255.0)
#                     processed_masks.append(augmented_mask / 255.0)
#             else:
#                 processed_images.append(resized_img / 255.0)
#                 processed_masks.append(resized_mask / 255.0)

#         return np.array(processed_images), np.array(processed_masks)
    
#     def _save_preprocessed_data(self, image, mask):
#         """Save the preprocessed images and masks in a different directory."""
#         if self.preprocessed_save_dir:
#             # Use a unique count for saving preprocessed data
#             image_filename = os.path.join(self.preprocessed_save_dir, "images", f"image_{self.augmentation_count}.png")
#             mask_filename = os.path.join(self.preprocessed_save_dir, "masks", f"mask_{self.augmentation_count}.png")
#             cv2.imwrite(image_filename, image)
#             cv2.imwrite(mask_filename, mask)
    
#     def _augment_and_save(self, image, mask, save_dir, count):
#         # Flip augmentation
#         if random.random() > 0.5:
#             image = cv2.flip(image, 1)
#             mask = cv2.flip(mask, 1)

#         if random.random() > 0.5:
#             image = cv2.flip(image, 0)
#             mask = cv2.flip(mask, 0)

#         # Rotation augmentation
#         angle = random.randint(-30, 30)
#         rows, cols = image.shape[:2]
#         rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#         image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
#         mask = cv2.warpAffine(mask, rotation_matrix, (cols, rows))

#         # Translation augmentation
#         max_trans_x = int(0.2 * cols)
#         max_trans_y = int(0.2 * rows)
#         trans_x = random.randint(-max_trans_x, max_trans_x)
#         trans_y = random.randint(-max_trans_y, max_trans_y)
#         translation_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
#         image = cv2.warpAffine(image, translation_matrix, (cols, rows))
#         mask = cv2.warpAffine(mask, translation_matrix, (cols, rows))

#         # Brightness augmentation
#         if random.random() > 0.5:
#             factor = 0.5 + random.random()  # Random factor between 0.5 and 1.5
#             image = np.clip(image * factor, 0, 255).astype(np.uint8)

#         # Gaussian noise augmentation
#         if random.random() > 0.5:
#             noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
#             image = cv2.add(image, noise)

#         # Save augmented images and masks
#         augmented_image_filename = os.path.join(save_dir, "images", f"image_aug_{count}.png")
#         augmented_mask_filename = os.path.join(save_dir, "masks", f"mask_aug_{count}.png")
#         cv2.imwrite(augmented_image_filename, image)
#         cv2.imwrite(augmented_mask_filename, mask)

#         return image, mask

