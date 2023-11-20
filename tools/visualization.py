import matplotlib.pyplot as plt
import PIL
import numpy as np


def plot_ship_statistics(feature):
    """
    Plot statistics for a binary feature indicating the presence of ships in images.

    Parameters:
    - feature (pandas.Series): A binary feature indicating the presence (True) or absence (False) of ships.

    Returns:
    None

    This function generates a bar plot and a pie chart to visualize the distribution and percentage
    of images with and without ships. It provides insights into the balance of the binary feature.
    """
    value_counts = feature.value_counts()

    # Define colors for True and False
    colors = ['red', 'green']

    # Specify the fraction of the radius to offset each wedge
    explode = (0.05, 0)  # Add 5% explosion (offset) to the first slice (True)

    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot 1: Bar plot
    bars = axs[0].bar(
        value_counts.index.astype(str),
        value_counts,
        color=colors,
    )

    # Add text labels at the top of each bar
    for bar in bars:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval + 500, int(yval), ha='center', va='bottom')

    # Set plot labels and title for subplot 1
    axs[0].set_xlabel('has_ship')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Amount of Images with and without Ships')

    # Subplot 2: Pie chart
    axs[1].pie(
        value_counts,
        labels=value_counts.index.astype(str),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=explode,
    )

    # Set plot title for subplot 2
    axs[1].set_title('Percentage of Images with and without Ships')

    # Adjust layout
    plt.tight_layout()

    # Display the subplots
    plt.show()


def get_image(image_name, directory):
    """
    Retrieve an image from a specified directory.

    Parameters:
    - image_name (str): The name of the image file.
    - directory (str): The directory containing the image file.

    Returns:
    numpy.ndarray: A NumPy array representing the RGB pixel values of the image.

    This function opens an image file using the Pillow (PIL) library, converts it to a NumPy array
    representing RGB pixel values, and returns the array.
    """
    # Open the image file using Pillow
    img = PIL.Image.open(f'{directory}/{image_name}')

    # Convert the image to a NumPy array representing RGB pixel values
    rgb_pixels = np.array(img)

    # Return the RGB pixel values array
    return rgb_pixels


def get_n_ship_image_name(n_ships, ship_counts, directory):
    """
    Retrieve the image name for an image with a specified number of ships.

    Parameters:
    - n_ships (int): The target number of ships.
    - ship_counts (pandas.DataFrame): DataFrame containing ship counts for each image.
    - directory (str): The directory containing the images.

    Returns:
    str: The name of an image with the specified number of ships.

    This function selects a random image name from the provided ship_counts DataFrame
    where the ship count matches the specified number of ships.
    """
    # Select a random image name with the specified number of ships
    image_name = ship_counts[ship_counts.ship_count == n_ships].sample().image_id.iloc[0]

    # Return the selected image name
    return image_name


def get_n_ship_image(n_ships, ship_counts, directory):
    """
    Retrieve the RGB pixel values for an image with a specified number of ships.

    Parameters:
    - n_ships (int): The target number of ships.
    - ship_counts (pandas.DataFrame): DataFrame containing ship counts for each image.
    - directory (str): The directory containing the images.

    Returns:
    numpy.ndarray: A NumPy array representing the RGB pixel values of the image.

    This function combines the functionality of retrieving an image name based on the specified
    number of ships and obtaining the RGB pixel values of the corresponding image.
    """
    # Retrieve the image name based on the specified number of ships
    image_name = get_n_ship_image_name(n_ships, ship_counts, directory)

    # Retrieve the RGB pixel values of the corresponding image
    image_rgb = get_image(image_name, directory)

    # Return the RGB pixel values array
    return image_rgb


def show_n_ships(n, ship_counts, directory):
    """
    Display a grid of images with a specified number of ships.

    Parameters:
    - n (int): The target number of ships.
    - ship_counts (pandas.DataFrame): DataFrame containing ship counts for each image.
    - directory (str): The directory containing the images.

    This function generates a grid of images with the specified number of ships and displays it using
    Matplotlib subplots.
    """
    # Generate a list of images with the specified number of ships
    images = [get_n_ship_image(n, ship_counts, directory) for i in range(6)]

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), tight_layout=True)

    # Set the overall title for the grid
    plt.suptitle(f'Images with {n} Ships', fontsize=18)

    # Display each image in the grid
    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            axes[i, j].imshow(images[idx], cmap='viridis')
            axes[i, j].axis('off')

    # Show the grid of images
    plt.show()


def rle_decode(rle, shape=(768, 768)):
    """
    Decode a run-length encoded (RLE) mask.

    Parameters:
    - rle (str): The run-length encoded mask as a string.
    - shape (tuple): The shape of the target mask (default is (768, 768)).

    Returns:
    numpy.ndarray: A NumPy array representing the decoded binary mask.

    This function decodes a run-length encoded mask and returns a binary mask with the specified shape.
    """
    # Convert the run-length encoded string to a list of integers
    nums = [int(num) for num in rle.split()]

    # Extract start positions and lengths from the list
    starts, lengths = np.array(nums[::2]), np.array(nums[1::2])

    # Calculate end positions based on starts and lengths
    ends = starts + lengths - 1

    # Initialize an array to represent the mask
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Set the corresponding elements in the mask array to 1 based on start and end positions
    for start, end in zip(starts, ends):
        mask[start:end + 1] = 1

    # Reshape the mask array to the specified shape
    return mask.reshape(shape).T


def get_image_masks(image_name, segmentations):
    """
    Retrieve the encoded masks for a specific image.

    Parameters:
    - image_name (str): The name of the target image.
    - segmentations (pandas.DataFrame): DataFrame containing segmentation information.

    Returns:
    numpy.ndarray: An array of encoded masks for the specified image.

    This function extracts the encoded masks for a particular image from the provided segmentations DataFrame.
    """
    # Filter segmentations DataFrame based on the image name
    masks = segmentations[segmentations['image_id'] == image_name]['encoded_pixels'].values

    # Return the array of encoded masks
    return masks


def image_from_masks(masks, shape=(768, 768)):
    """
    Generate a binary image from a list of encoded masks.

    Parameters:
    - masks (str or list): The encoded masks as a string or list of strings.
    - shape (tuple): The shape of the target image (default is (768, 768)).

    Returns:
    numpy.ndarray: A binary image generated from the decoded masks.

    This function decodes a list of run-length encoded masks and creates a binary image based on the specified shape.
    """
    # Initialize an array to represent the masks image
    masks_image = np.zeros(shape, dtype=np.uint8)

    # Check if masks is a single string and decode it
    if isinstance(masks, str) and len(masks) == 1:
        masks_image += rle_decode(masks, shape)
        return np.expand_dims(masks_image, -1)

    # Iterate through the list of masks and decode each one
    for mask in masks:
        if isinstance(mask, str):
            masks_image += rle_decode(mask, shape)

    # Add an extra dimension to the masks image for compatibility
    return np.expand_dims(masks_image, -1)


def get_image_mask_combination(image, mask):
    """
    Combine an image and its corresponding mask.

    Parameters:
    - image (numpy.ndarray): The original image.
    - mask (numpy.ndarray): The binary mask.

    Returns:
    numpy.ndarray: An image with the mask applied.

    This function combines the original image and its corresponding binary mask, applying a specified color to the masked regions.
    """
    # Specify the color to use for the masked regions
    color = [255, 0, 0]

    # Create a mask for regions where the mask is not applied
    non_masked = mask[:, :, 0] != 1

    # Apply the mask and set the color only in the masked regions
    masked_array = np.where(mask == 1, color, image)

    # Set the color to the original image values where the mask is not applied
    masked_array[non_masked] = image[non_masked]

    return masked_array


def plot_n_ship_image_mask(ships_amount, ship_counts, segmentations, directory):
    """
    Plot the original image, binary mask, and masked image for a specific number of ships.

    Parameters:
    - ships_amount (int): The desired number of ships.
    - ship_counts (pd.DataFrame): DataFrame containing ship counts for each image.
    - segmentations (pd.DataFrame): DataFrame containing encoded masks for each image.
    - directory (str): The directory containing the images.

    Returns:
    None

    This function plots the original image, binary mask, and masked image for a randomly selected image with a specific number of ships.
    """
    # Get a random image name with the specified number of ships
    image_name = get_n_ship_image_name(ships_amount, ship_counts, directory)
    print(f'⛳️ Image name: {image_name}\n⛳️ Amount of ships: {ships_amount}')

    # Load the original image
    image_original = get_image(image_name, directory)

    # Get masks for the image
    masks = get_image_masks(image_name, segmentations)
    image_masks = image_from_masks(masks)

    # Combine the original image and masks
    combined = get_image_mask_combination(image_original, image_masks)

    # Plot the images and masks
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    ax[0].imshow(image_original)
    ax[1].imshow(image_masks[:, :, 0], cmap='gray')  # Display only the first channel of the binary mask
    ax[2].imshow(combined)

    ax[0].set_title('Original Image')
    ax[1].set_title('Binary Mask')
    ax[2].set_title('Masked Image')

    plt.show()


def plot_loss(history):
    """
    Plot training and validation loss over epochs.

    Parameters:
    - history (dict): Dictionary containing 'loss' and 'val_loss' keys.

    Returns:
    None
    """
    # Create a subplot with 1 row and 1 column, setting the figure size
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Plot the training loss
    ax.plot(history['loss'], label="train loss")

    # Plot the validation loss
    ax.plot(history['val_loss'], label="validation loss")

    # Set title and labels for the plot
    ax.set_title("Loss")
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Loss")

    # Add legend to distinguish between training and validation loss
    ax.legend()

    # Display the plot
    plt.show()


def plot_dice_coefficient(history):
    """
    Plot training and validation dice coefficient over epochs.

    Parameters:
    - history (dict): Dictionary containing 'dice_coefficient' and 'val_dice_coefficient' keys.

    Returns:
    None
    """
    # Create a subplot with 1 row and 1 column, setting the figure size
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Plot the training dice coefficient
    ax.plot(history['dice_coefficient'], label="train dice coefficient")

    # Plot the validation dice coefficient
    ax.plot(history['val_dice_coefficient'], label="validation dice coefficient")

    # Set title and labels for the plot
    ax.set_title("Dice Coefficient")
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Dice Coefficient")

    # Add legend to distinguish between training and validation dice coefficient
    ax.legend()

    # Display the plot
    plt.show()