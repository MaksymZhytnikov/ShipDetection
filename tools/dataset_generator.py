from skimage.io import imread
from skimage.transform import resize 
import numpy as np
import os
from tools.visualization import image_from_masks


def get_image_generator(in_df, batch_size, target_size, directory):
    """
    Generate batches of resized images and masks from a DataFrame.

    Parameters:
    - in_df (pd.DataFrame): DataFrame containing image information and segmentation masks.
    - batch_size (int): Size of the batches to generate.
    - target_size (tuple): Tuple specifying the target size for resizing (height, width).
    - directory (str): Directory containing the images.

    Returns:
    generator: A generator that yields batches of resized images and masks.

    This function generates batches of resized images and masks for training a neural network.
    """
    # Group DataFrame by image_id
    all_batches = list(in_df.groupby('image_id'))
    out_rgb = []
    out_mask = []

    while True:
        np.random.shuffle(all_batches)

        for c_img_id, c_masks in all_batches:
            # Load the original image and masks
            rgb_path = os.path.join(directory, c_img_id)
            c_img = imread(rgb_path)
            c_mask = image_from_masks(c_masks['encoded_pixels'].values)

            # Resize the images and masks
            c_img_resized = resize(c_img, target_size)
            c_mask_resized = resize(c_mask, target_size, mode='constant', anti_aliasing=False)

            # Append the resized images and masks to the output lists
            out_rgb += [c_img_resized]
            out_mask += [c_mask_resized]

            # Yield batches when the specified batch size is reached
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb).astype(np.float32), np.stack(out_mask).astype(np.float32)
                out_rgb, out_mask = [], []


def augment(data_generator, X_generator, y_generator, seed=42):
    """
    Augment images and masks using data generators.

    Parameters:
    - data_generator: Generator providing original images and masks.
    - X_generator: Image data generator for augmentation of X (images).
    - y_generator: Image data generator for augmentation of y (masks).
    - seed (int): Seed for random number generation.

    Returns:
    generator: A generator that yields augmented images and masks.

    This function takes an existing data generator along with separate generators for X and y
    and yields batches of augmented images and masks.
    """
    np.random.seed(seed)

    for X, y in data_generator:
        # Create augmented images
        augmented_images = X_generator.flow(
            255 * X,
            batch_size=X.shape[0],
            seed=seed,
            shuffle=True,
        )

        # Create augmented masks
        augmented_masks = y_generator.flow(
            y,
            batch_size=X.shape[0],
            seed=seed,
            shuffle=True,
        )

        # Yield batches of augmented images and masks
        yield next(augmented_images) / 255.0, next(augmented_masks)


