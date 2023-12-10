# Problem Statement For 2nd Proxmed-LPU Hackathon
Problem Description:
The objective of this hackathon challenge is to develop a robust and efficient algorithm or AI model capable of accurately segmenting the hypodense region from Brain Non-Contrast Computed Tomography (NCCT) images, regardless of the slice thickness and orientation of the images. The primary goal is to automate and streamline the identification of early ischemic changes in acute stroke patients.

## Data Loading and Normalization:

The code is designed to work with medical imaging data stored in NIFTI format (.nii.gz).
The load_and_normalize_data function loads the Non-Contrast CT (NCCT) image and its corresponding Region of Interest (ROI) mask for a given case_id.
The NCCT image is loaded, and its intensity values are normalized to the range [0, 1].
The ROI mask is also loaded as a separate image and converted from float32 to unsigned integer.

## Data Visualization:

The visualize_all_slices function is responsible for visualizing slices of the NCCT image and the ROI mask side by side.
For a specified case_id, it iterates through the slices of the 3D image and creates visualizations for each slice.
Three subplots are created for each slice: one for the normalized NCCT image, one for the binary ROI mask, and one for the overlay of the image and mask.
This visualization helps in understanding the content of the data and the correspondence between the NCCT image and its ROI mask.

## Resizing and Padding:

Another function (resize_and_pad_images) is defined for resizing and padding the images to a target size and a target number of slices.
The goal is to ensure uniformity in the dimensions of the images, which is important for training deep learning models.
It uses the resize function from the skimage library to resize the images and a custom pad_slices function to handle padding or truncation of slices.
Saving as .npy Files:

The save_as_npy function is responsible for saving the normalized, resized, and padded NCCT images and ROI masks as NumPy (.npy) files in their corresponding folders.
These saved files can be used for subsequent training of machine learning models.

from skimage.transform import resize


data_folder = "/content/HYPODENSITY-DATA"

## Function to load and normalize the NIFTI files
def load_and_normalize_data(case_id):
    # Load NCCT image
    ncct_path = os.path.join(data_folder, case_id, f"{case_id}_NCCT.nii.gz")
    ncct_img = nib.load(ncct_path).get_fdata()

    # Load ROI mask
    roi_path = os.path.join(data_folder, case_id, f"{case_id}_ROI.nii.gz")
    roi_mask = nib.load(roi_path).get_fdata()

    # Normalize the NCCT image to [0, 1]
    ncct_img_normalized = (ncct_img - np.min(ncct_img)) / (np.max(ncct_img) - np.min(ncct_img))

    # Convert ROI mask to binary format
    roi_mask_binary = (roi_mask > 0.5
                       ).astype(np.uint8)

    return ncct_img_normalized, roi_mask_binary

## Function to resize and pad images
def resize_and_pad_images(ncct_img_normalized, roi_mask_binary, target_size, target_slices):
    # Resize images
    ncct_img_resized = resize(ncct_img_normalized, target_size, anti_aliasing=True)
    roi_mask_resized = resize(roi_mask_binary, target_size, anti_aliasing=False, preserve_range=True).astype(np.uint8)

    # Pad or truncate slices
    ncct_img_resized_padded = pad_slices(ncct_img_resized, target_slices)
    roi_mask_resized_padded = pad_slices(roi_mask_resized, target_slices)

    return ncct_img_resized_padded, roi_mask_resized_padded

## Function to save images as .npy files in their corresponding folders
def save_as_npy(case_id, ncct_img_resized, roi_mask_resized):
    np.save(os.path.join(data_folder, case_id, f"{case_id}_NCCT.npy"), ncct_img_resized)
    np.save(os.path.join(data_folder, case_id, f"{case_id}_ROI.npy"), roi_mask_resized)

## Function to pad slices
def pad_slices(volume, target_slices):
    current_slices = volume.shape[2]

    if current_slices < target_slices:
        pad_width = ((0, 0), (0, 0), (0, target_slices - current_slices))
        padded_volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
        return padded_volume
    elif current_slices > target_slices:
        # Truncate slices if there are more than the target
        truncated_volume = volume[:, :, :target_slices]
        return truncated_volume
    else:
        return volume


## Function to visualize all slices
def visualize_all_slices(case_id, ncct_img, roi_mask):
    num_slices = ncct_img.shape[2]

    for slice_idx in range(num_slices):
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(ncct_img[:, :, slice_idx], cmap='gray')
        plt.title(f'Normalized NCCT Image - {case_id} - Slice {slice_idx}')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(roi_mask[:, :, slice_idx], cmap='viridis', alpha=0.7)
        plt.title(f'Binary ROI Mask - {case_id} - Slice {slice_idx}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(ncct_img[:, :, slice_idx], cmap='gray')
        plt.imshow(roi_mask[:, :, slice_idx], cmap='viridis', alpha=0.5)
        plt.title(f'Overlay - {case_id} - Slice {slice_idx}')
        plt.axis('off')

        plt.show()

## Set the target size and number of slices
target_size = (128,128)
target_slices = 128

## Iterate through all case IDs, load, normalize, resize, pad, and save as .npy files in their corresponding folders
for case_id in case_ids:
    ncct_img_normalized, roi_mask_binary = load_and_normalize_data(case_id)
    ncct_img_resized_padded, roi_mask_resized_padded = resize_and_pad_images(ncct_img_normalized, roi_mask_binary, target_size, target_slices)
    visualize_all_slices(case_id, ncct_img_resized_padded, roi_mask_resized_padded)
    save_as_npy(case_id, ncct_img_resized_padded, roi_mask_resized_padded)

## Data Augmentation:

Data augmentation is performed to increase the diversity of the training dataset. This includes random rotation, horizontal and vertical flips, addition of Gaussian noise, and random adjustment of image contrast.
The augmented images and masks are saved in separate folders with an index indicating the augmentation.
import os
import numpy as np
from skimage.transform import rotate
from skimage import img_as_ubyte
from skimage.util import random_noise
from skimage import exposure

data_folder = "/content/HYPODENSITY-DATA"

## Function to load .npy files
def load_npy_files(case_id):
    ncct_path = os.path.join(data_folder, case_id, f"{case_id}_NCCT.npy")
    roi_path = os.path.join(data_folder, case_id, f"{case_id}_ROI.npy")

    ncct_img = np.load(ncct_path)
    roi_mask = np.load(roi_path)

    return ncct_img, roi_mask


## Function for data augmentation
def augment_data(images, masks, num_samples=25):
    augmented_images = []
    augmented_masks = []

    for _ in range(num_samples):
        # Random rotation (between -25 and 25 degrees)
        angle = np.random.uniform(low=-25, high=25)
        rotated_images = rotate(images, angle, preserve_range=True)
        rotated_masks = rotate(masks, angle, preserve_range=True)

        # Random horizontal flip
        if np.random.choice([True, False]):
            rotated_images = np.fliplr(rotated_images)
            rotated_masks = np.fliplr(rotated_masks)

        # Random vertical flip
        if np.random.choice([True, False]):
            rotated_images = np.flipud(rotated_images)
            rotated_masks = np.flipud(rotated_masks)

        # Add Gaussian noise
        std = np.random.uniform(0.0, 0.08)
        noisy_images = random_noise(rotated_images, var=std**2)

        # Adjust image contrast randomly
        if np.random.choice([True, False]):
            noisy_images = exposure.adjust_gamma(noisy_images, gamma=np.random.uniform(0.5, 1.5))

        augmented_images.append(noisy_images)
        augmented_masks.append(rotated_masks.astype(np.uint8))

    return augmented_images, augmented_masks

## Function to save images as .npy files 
def save_as_npy(case_id, ncct_img_resized, roi_mask_resized, i, augmented_index):
    output_folder = os.path.join(data_folder, f"{case_id}_aug_{augmented_index}")
    os.makedirs(output_folder, exist_ok=True)  # Ensure the directory exists

    np.save(os.path.join(output_folder, f"{case_id}_NCCT_aug_{i}.npy"), ncct_img_resized)
    np.save(os.path.join(output_folder, f"{case_id}_ROI_aug_{i}.npy"), roi_mask_resized)

## Iterate through all case IDs, load, augment, and save as .npy files in their corresponding folders
for case_id in case_ids:
    # Load training data for augmentation
    ncct_img, roi_mask = load_npy_files(case_id)

    # Perform data augmentation
    augmented_ncct, augmented_roi_mask = augment_data(ncct_img, roi_mask)

    # Save augmented data
    for i, (aug_ncct, aug_roi_mask) in enumerate(zip(augmented_ncct, augmented_roi_mask)):
        save_as_npy(case_id, aug_ncct, aug_roi_mask, i, augmented_index=i)


## Verification:

The code includes a verification section where it loads and verifies the sizes of the original and augmented images and masks for a few case IDs.
This step ensures that the preprocessing and augmentation processes have been applied correctly.
import os
import numpy as np
import matplotlib.pyplot as plt


data_folder = "/content/HYPODENSITY-DATA"

## Function to load augmented .npy files
def load_augmented_npy_files(case_id, aug_index):
    ncct_path = os.path.join(data_folder, f"{case_id}_aug_{aug_index}", f"{case_id}_NCCT_aug_{aug_index}.npy")
    roi_path = os.path.join(data_folder, f"{case_id}_aug_{aug_index}", f"{case_id}_ROI_aug_{aug_index}.npy")

    ncct_img = np.load(ncct_path)
    roi_mask = np.load(roi_path)

    return ncct_img, roi_mask

## Choose a case and its corresponding augmentation index for visualization
case_id_for_visualization = "ProxmedImg014"
augmentation_index_for_visualization = 9


## Load and visualize the augmented data
ncct_img_augmented, roi_mask_augmented = load_augmented_npy_files(case_id_for_visualization, augmentation_index_for_visualization)

## Visualize all slices
num_slices = 10


plt.figure(figsize=(18, 6 * num_slices))

for slice_idx in range(num_slices):
    plt.subplot(num_slices, 2, 2 * slice_idx + 1)
    plt.imshow(ncct_img_augmented[:, :, slice_idx], cmap='gray')
    plt.title(f'Augmented NCCT Image - {case_id_for_visualization} - Augmentation Index {augmentation_index_for_visualization} - Slice {slice_idx}')
    plt.axis('off')

    plt.subplot(num_slices, 2, 2 * slice_idx + 2)
    plt.imshow(roi_mask_augmented[:, :, slice_idx], cmap='viridis', alpha=0.7)
    plt.title(f'Augmented ROI Mask - {case_id_for_visualization} - Augmentation Index {augmentation_index_for_visualization} - Slice {slice_idx}')
    plt.axis('off')

plt.show()
import os
import numpy as np

## Set the path to the data folder
data_folder = "/content/HYPODENSITY-DATA"

## Function to load normalized resized .npy files
def load_normalized_resized_npy(case_id):
    ncct_path = os.path.join(data_folder, f"{case_id}", f"{case_id}_NCCT.npy")
    roi_path = os.path.join(data_folder, f"{case_id}", f"{case_id}_ROI.npy")

    ncct_img = np.load(ncct_path)
    roi_mask = np.load(roi_path)

    return ncct_img, roi_mask

## Function to load augmented .npy files
def load_augmented_npy_files(case_id, aug_index):
    ncct_path = os.path.join(data_folder, f"{case_id}_aug_{aug_index}", f"{case_id}_NCCT_aug_{aug_index}.npy")
    roi_path = os.path.join(data_folder, f"{case_id}_aug_{aug_index}", f"{case_id}_ROI_aug_{aug_index}.npy")

    ncct_img = np.load(ncct_path)
    roi_mask = np.load(roi_path)

    return ncct_img, roi_mask

## Iterate through all case IDs to load and verify sizes
case_ids = ["ProxmedImg006", "ProxmedImg013", "ProxmedImg014", "ProxmedImg021", 
            "ProxmedImg022", "ProxmedImg025", "ProxmedImg043", "ProxmedImg331"]

for case_id in case_ids:
    # Load and verify the size of the original normalized resized NCCT image
    ncct_img_original, roi_mask_original = load_normalized_resized_npy(case_id)
    print(f"Size of the original normalized resized NCCT image for {case_id}: {ncct_img_original.shape}")
    print(f"Size of the corresponding ROI mask for {case_id}: {roi_mask_original.shape}")

    # Load and verify the size of the augmented NCCT image (choose an augmentation index, e.g., 0)
    augmentation_index_for_verification = 5
    ncct_img_augmented, roi_mask_augmented = load_augmented_npy_files(case_id, augmentation_index_for_verification)
    print(f"Size of the augmented NCCT image for {case_id}, Augmentation {augmentation_index_for_verification}: {ncct_img_augmented.shape}")
    print(f"Size of the corresponding augmented ROI mask for {case_id}, Augmentation {augmentation_index_for_verification}: {roi_mask_augmented.shape}")

## Size of the original normalized resized NCCT image for ProxmedImg006: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg006: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg006, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg006, Augmentation 5: (128, 128, 128)
Size of the original normalized resized NCCT image for ProxmedImg013: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg013: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg013, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg013, Augmentation 5: (128, 128, 128)
Size of the original normalized resized NCCT image for ProxmedImg014: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg014: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg014, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg014, Augmentation 5: (128, 128, 128)
Size of the original normalized resized NCCT image for ProxmedImg021: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg021: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg021, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg021, Augmentation 5: (128, 128, 128)
Size of the original normalized resized NCCT image for ProxmedImg022: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg022: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg022, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg022, Augmentation 5: (128, 128, 128)
Size of the original normalized resized NCCT image for ProxmedImg025: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg025: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg025, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg025, Augmentation 5: (128, 128, 128)
Size of the original normalized resized NCCT image for ProxmedImg043: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg043: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg043, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg043, Augmentation 5: (128, 128, 128)
Size of the original normalized resized NCCT image for ProxmedImg331: (128, 128, 128)
Size of the corresponding ROI mask for ProxmedImg331: (128, 128, 128)
Size of the augmented NCCT image for ProxmedImg331, Augmentation 5: (128, 128, 128)
Size of the corresponding augmented ROI mask for ProxmedImg331, Augmentation 5: (128, 128, 128)


## Data Splitting and Copying:

The dataset is split into training and testing sets using train_test_split from scikit-learn.
The function copy_files is designed to copy the original and augmented data for each case ID into separate folders for training and testing.
Both the NCCT images and their corresponding ROI masks are copied.
import os
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile

## Set the path to the data folder
data_folder = "/content/HYPODENSITY-DATA"

## Output folders for training and testing data
output_folder_train = "/content/training_data"
output_folder_test = "/content/testing_data"

## Function to load normalized resized .npy files
def load_normalized_resized_npy(case_id):
    ncct_path = os.path.join(data_folder, f"{case_id}", f"{case_id}_NCCT.npy")
    roi_path = os.path.join(data_folder, f"{case_id}", f"{case_id}_ROI.npy")

    ncct_img = np.load(ncct_path)
    roi_mask = np.load(roi_path)

    return ncct_img, roi_mask

## Function to load augmented .npy files
def load_augmented_npy_files(case_id, aug_index):
    ncct_path = os.path.join(data_folder, f"{case_id}_aug_{aug_index}", f"{case_id}_NCCT_aug_{aug_index}.npy")
    roi_path = os.path.join(data_folder, f"{case_id}_aug_{aug_index}", f"{case_id}_ROI_aug_{aug_index}.npy")

    ncct_img = np.load(ncct_path)
    roi_mask = np.load(roi_path)

    return ncct_img, roi_mask

## Function to copy files
def copy_files(case_ids, output_folder):
    for case_id in case_ids:
        try:
            # Create subfolders for images and masks
            os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_folder, 'masks'), exist_ok=True)

            ncct_img, roi_mask = load_normalized_resized_npy(case_id)
            np.save(os.path.join(output_folder, 'images', f"{case_id}_NCCT.npy"), ncct_img)
            np.save(os.path.join(output_folder, 'masks', f"{case_id}_ROI.npy"), roi_mask)

            # Copy augmented data
            aug_index = 0
            while True:
                try:
                    ncct_img_aug, roi_mask_aug = load_augmented_npy_files(case_id, aug_index)
                    np.save(os.path.join(output_folder, 'images', f"{case_id}_NCCT_aug_{aug_index}.npy"), ncct_img_aug)
                    np.save(os.path.join(output_folder, 'masks', f"{case_id}_ROI_aug_{aug_index}.npy"), roi_mask_aug)
                    aug_index += 1
                except FileNotFoundError:
                    # Break the loop if the file is not found
                    break
        except FileNotFoundError:
            print(f"Error loading data for case ID: {case_id}")

## List of case IDs for training and testing
all_case_ids = ["ProxmedImg006", "ProxmedImg013", "ProxmedImg014", "ProxmedImg021", 
                "ProxmedImg022", "ProxmedImg025", "ProxmedImg043", "ProxmedImg331"]

## Split the data into training and testing sets
train_case_ids, test_case_ids = train_test_split(all_case_ids, test_size=0.1, random_state=42)

## Create output folders
os.makedirs(output_folder_train, exist_ok=True)
os.makedirs(output_folder_test, exist_ok=True)

## Copy training data
copy_files(train_case_ids, output_folder_train)

## Copy testing data
copy_files(test_case_ids, output_folder_test)

## Data Loading and Sorting:

The load_img function is defined to load the images from a specified directory (img_dir) based on a provided list of image names (img_list).
A custom imageLoader function is defined for creating a generator that yields batches of images and masks.
This generator is designed to handle loading a specified batch size of images and masks from the training dataset.
Images and masks are sorted to ensure consistency in pairing during loading.

## Generator for Training Data:

The paths to the training images (train_img_dir) and masks (train_mask_dir) are specified.
The generator (train_img_datagen) is created using the imageLoader function, specifying the paths and batch size.
The generator yields batches of images and masks during training.

  import random

  def load_img(img_dir, img_list):
      images = []
      for i, image_name in enumerate(img_list):
          if image_name.endswith('.npy'):
              image = np.load(os.path.join(img_dir, image_name))
              images.append(image)
      images = np.array(images)
      return images

  def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
      img_list.sort()  # Ensure images are sorted
      mask_list.sort()  # Ensure masks are sorted
      L = len(img_list)

      while True:
          batch_start = 0
          batch_end = batch_size

          while batch_start < L:
              limit = min(batch_end, L)
              print(f"Loading images: {img_list[batch_start:limit]}")
              print(f"Loading masks: {mask_list[batch_start:limit]}")
              X = load_img(img_dir, img_list[batch_start:limit])

              # Ensure that images have the shape (batch_size, 128, 128, 128, 1)
              X = np.expand_dims(X, axis=-1)  # Assuming your images are grayscale

              Y = load_img(mask_dir, mask_list[batch_start:limit])

              # Ensure that masks have the shape (batch_size, 128, 128, 128, 1)
              Y = np.expand_dims(Y, axis=-1)

              yield (X, Y)

              batch_start += batch_size
              batch_end += batch_size



  ## Your paths
  train_img_dir = "/content/training_data/images"
  train_mask_dir = "/content/training_data/masks"

  ## List of file names
  train_img_list = os.listdir(train_img_dir)
  train_mask_list = os.listdir(train_mask_dir)

  batch_size = 3


  ## Use your paths in the generator
  train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                  train_mask_dir, train_mask_list, batch_size)

  ## Print the shape of a few batches
  for i in range(3):  # Change 3 to the number of batches you want to check
      img, msk = train_img_datagen.__next__()

      print(f"Batch {i + 1}:")
      print("Images shape:", img.shape)
      print("Masks shape:", msk.shape)
      print("\n")


## Visualization of Batches:

The visualize_batch function is defined to visualize a few samples from each batch.
For each batch, it displays a slice of both the image and the corresponding mask for multiple samples.
def visualize_batch(images, masks, slice_num=25):
    num_samples = images.shape[0]

    for i in range(num_samples):
        plt.figure(figsize=(12, 6))

        # Display the image slice
        plt.subplot(1, 2, 1)
        plt.imshow(images[i, ..., slice_num, 0], cmap='gray')  # Assuming images are grayscale
        plt.title('Image')

        # Display the mask slice
        plt.subplot(1, 2, 2)
        plt.imshow(masks[i, ..., slice_num, 0], cmap='viridis')  # Assuming masks are binary
        plt.title('Mask')

        plt.show()

## Visualize a few batches
for i in range(5):  # Change 3 to the number of batches you want to visualize
    img_batch, mask_batch = train_img_datagen.__next__()
    visualize_batch(img_batch, mask_batch)


## Dice Coefficient Function (dice_coefficient):

Computes the Dice coefficient, a metric commonly used in segmentation tasks.
It measures the similarity between the predicted and true segmentation masks.

## Dice Loss Function (dice_loss):

Defines the Dice loss as 1 minus the Dice coefficient.
This is used as the loss function during model training.

# U-Net Model Architecture (unet_model):
Input:

The model takes 3D volumetric data as input, with dimensions (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS).
## Contracting (Encoder) Path:

Consists of several convolutional blocks with 3D convolutions, each followed by ReLU activation and dropout.
Max pooling layers reduce the spatial dimensions.
## Bottleneck:

A convolutional block with 3D convolutions.
## Expanding (Decoder) Path:

Consists of transpose convolutions to upsample the feature maps.
Concatenation with the corresponding feature maps from the contracting path.
Convolutional blocks similar to the contracting path but without the max pooling.
## Output:

The final layer consists of a 3D convolution with a sigmoid activation function, outputting a binary mask.
## Model Compilation:

Adam optimizer is used with a learning rate of 0.0001.
The model is compiled using the custom Dice loss and Dice coefficient as metrics.
## Training Process:
## Data Loading:

Images and masks are loaded using a custom generator (imageLoader).
The generator yields batches of 3D images and masks.
## Training Loop:

The model is trained using a loop over a specified number of epochs.
For each epoch, the training data generator is reset.
The model is trained on each batch, and the loss is printed for every 10 batches.
## Model Saving:
import os
import numpy as np
import random
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout
from tensorflow.keras.optimizers import Adam

## Define Dice coefficient function
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(K.cast(y_true, 'float32'))  # Cast y_true to float32
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

## Define Dice loss function
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

## U-Net model

kernel_initializer = 'he_uniform'

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    # Expansive path
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()

    # Compile the model with your custom Dice Loss and Dice Coefficient
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=dice_loss,
                  metrics=[dice_coefficient])

    return model

## Paths to your training data
train_img_dir = "/content/training_data/images"
train_mask_dir = "/content/training_data/masks"

## List of file names
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

## Batch size
batch_size = 3

def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.endswith('.npy'):
            image = np.load(os.path.join(img_dir, image_name))
            images.append(image)
    images = np.array(images)
    return images

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
      img_list.sort()  # Ensure images are sorted
      mask_list.sort()  # Ensure masks are sorted
      L = len(img_list)

      while True:
          batch_start = 0
          batch_end = batch_size

          while batch_start < L:
              limit = min(batch_end, L)
              print(f"Loading images: {img_list[batch_start:limit]}")
              print(f"Loading masks: {mask_list[batch_start:limit]}")
              X = load_img(img_dir, img_list[batch_start:limit])

              # Ensure that images have the shape (batch_size, 128, 128, 128, 1)
              X = np.expand_dims(X, axis=-1)  # Assuming your images are grayscale

              Y = load_img(mask_dir, mask_list[batch_start:limit])

              # Ensure that masks have the shape (batch_size, 128, 128, 128, 1)
              Y = np.expand_dims(Y, axis=-1)

              yield (X, Y)

              batch_start += batch_size
              batch_end += batch_size
## Use your paths in the generator
train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

## Create and train the model
model = simple_unet_model(128, 128, 128, 1, 1)

## Train the model
epochs = 20  # Adjust the number of epochs as needed

## Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Reset the generator for each epoch
    train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                    train_mask_dir, train_mask_list, batch_size)

    # Train for each batch
    for batch in range(len(train_img_list) // batch_size):
        img, msk = train_img_datagen.__next__()

        # Train the model on the current batch
        loss = model.train_on_batch(img, msk)

        # Print the loss for every 10 batches
        if batch % 10 == 0:
            print(f"Batch {batch}/{len(train_img_list) // batch_size}, Loss: {loss}")

## Save the trained model
model.save("trained_model.h5")
print("Training complete. Model saved.")

After training, the model is saved to a file named "trained_model.h5".
