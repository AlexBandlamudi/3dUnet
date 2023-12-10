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
## Data Augmentation:

Data augmentation is performed to increase the diversity of the training dataset. This includes random rotation, horizontal and vertical flips, addition of Gaussian noise, and random adjustment of image contrast.
The augmented images and masks are saved in separate folders with an index indicating the augmentation.
## Verification:

The code includes a verification section where it loads and verifies the sizes of the original and augmented images and masks for a few case IDs.
This step ensures that the preprocessing and augmentation processes have been applied correctly.

## Data Splitting and Copying:

The dataset is split into training and testing sets using train_test_split from scikit-learn.
The function copy_files is designed to copy the original and augmented data for each case ID into separate folders for training and testing.
Both the NCCT images and their corresponding ROI masks are copied.

## Data Loading and Sorting:

The load_img function is defined to load the images from a specified directory (img_dir) based on a provided list of image names (img_list).
A custom imageLoader function is defined for creating a generator that yields batches of images and masks.
This generator is designed to handle loading a specified batch size of images and masks from the training dataset.
Images and masks are sorted to ensure consistency in pairing during loading.

## Generator for Training Data:

The paths to the training images (train_img_dir) and masks (train_mask_dir) are specified.
The generator (train_img_datagen) is created using the imageLoader function, specifying the paths and batch size.
The generator yields batches of images and masks during training.

## Visualization of Batches:

The visualize_batch function is defined to visualize a few samples from each batch.
For each batch, it displays a slice of both the image and the corresponding mask for multiple samples.

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

After training, the model is saved to a file named "trained_model.h5".
