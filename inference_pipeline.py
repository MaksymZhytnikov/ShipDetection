import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from skimage.transform import resize 

from tools.dice import dice_loss, dice_coefficient
from tools.visualization import plot_n_ship_image_mask, get_n_ship_image_name, get_image, get_image_masks, image_from_masks

# Set the title of the Streamlit app
st.title('🛳️ Ship Detector')

# Header for the data visualization section
st.header('👨🏻‍🎨 Data Visualization')

# Cache data loading functions to improve performance
@st.cache_data()
def read_data():
    st.write('💾 Reading Data...')

    TRAIN_DIR = 'data/train_v2'
    TEST_DIR = 'data/test_v2'

    # Get a list of file names in the train and test directories
    train_names = os.listdir(TRAIN_DIR)
    test_names = os.listdir(TEST_DIR)

    # Read the CSV file containing ship segmentations
    segmentations = pd.read_csv('data/train_ship_segmentations_v2.csv')

    # Rename the columns for better readability
    segmentations.columns = ['image_id', 'encoded_pixels']

    # Create a new column 'has_ship' based on the presence of non-null values in the 'encoded_pixels' column
    segmentations['has_ship'] = segmentations.encoded_pixels.notnull()

    # Group the 'segmentations' DataFrame by 'image_id' and count the occurrences of 'has_ship' for each image
    ship_counts = segmentations.groupby('image_id')['has_ship'].agg('sum').reset_index()

    # Rename the columns for better readability
    ship_counts.columns = [ship_counts.columns[0], 'ship_count']

    # Calculate and add a new column 'file_size_kb' representing the file size in kilobytes for each image
    ship_counts['file_size_kb'] = ship_counts['image_id'].map(
        lambda c_img_id: os.stat(os.path.join(TRAIN_DIR, c_img_id)).st_size / 1024)

    # Filter out images with a file size less than or equal to 50 KB
    ship_counts = ship_counts[ship_counts['file_size_kb'] > 50]

    st.write('✅ Done!')    

    return segmentations, ship_counts, TRAIN_DIR, TEST_DIR


@st.cache_data()
def get_model():
    st.write('🪝 Loading Model...')
    model = load_model(
        'models/unet_model.h5',
        custom_objects={
            'dice_loss': dice_loss,
            'dice_coefficient': dice_coefficient,
        },
        compile=False,
    )
    model.compile()
    st.write('✅ Done!')  
    return model 


# Call the data loading functions
segmentations, ship_counts, TRAIN_DIR, TEST_DIR = read_data()

# Dropdown to select the amount of ships for data visualization
ships_amount_to_plot = st.selectbox(
   "Amount of ships on image?",
   [*range(16)],
   index=5,
   placeholder="Select amount...",
)

st.write(f'👨🏻‍🎨 Plotting image with {ships_amount_to_plot} ships...')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Call the plot_n_ship_image_mask function to display the image
fig_n_ships = plot_n_ship_image_mask(ships_amount_to_plot, ship_counts, segmentations, TRAIN_DIR)

# Display the image using pyplot
st.pyplot(fig_n_ships)

# Divider and header for model loading section
st.divider()
st.header('🤖 Model Loading')

# Call the function to load the model
model = get_model()

# Divider and header for model inference section
st.divider()
st.header('🚀 Model Inference')

# Dropdown to select the amount of ships for model inference
ships_amount = st.selectbox(
   "Amount of ships on image to predict?",
   [*range(16)],
   index=5,
   placeholder="Select amount...",
)

st.write(f'🗄️ Selecting image with {ships_amount} ships...')

# Get the image name with the selected amount of ships
image_n_ships_name = get_n_ship_image_name(ships_amount, ship_counts, TRAIN_DIR)

st.write(f'📝 Selected image id: {image_n_ships_name}')

st.write(f'🔮 Predicting image with {ships_amount} ships...')

# Get the RGB image and masks for the selected image
image_rgb = get_image(image_n_ships_name, TRAIN_DIR)
masks = get_image_masks(image_n_ships_name, segmentations)

# Create an image from masks for visualization
image_masks = image_from_masks(masks)

# Resize the original image and masks for model prediction
image_n_ships_resized = resize(image_rgb, (64,64,3))
image_masks_resized = resize(image_masks, (64,64,1))

# Perform model prediction
prediction = model.predict(np.array([image_n_ships_resized]))[0]

# Create subplots for visualization
fig_resized, ax = plt.subplots(1,3,figsize=(12,4))

# Display the resized original image, ground truth masks, and model prediction
ax[0].imshow(image_n_ships_resized)
ax[1].imshow(image_masks_resized)
ax[2].imshow(prediction)

# Set titles for subplots
ax[0].set_title('Original Resized Image')
ax[1].set_title('Ground Truth')
ax[2].set_title('Prediction')

# Display the subplots using pyplot
st.pyplot(fig_resized)

# Stop Streamlit button
if st.button("Stop Streamlit"):
    st.success('✅ App finished successfully!')
    st.balloons()
    st.stop()