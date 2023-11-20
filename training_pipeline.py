# Import necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
)
from tools.dataset_generator import get_image_generator, augment
from tools.unet import unet_model
from tools.dice import dice_coefficient, dice_loss
from tools.visualization import plot_loss, plot_dice_coefficient
import warnings
warnings.filterwarnings('ignore')

print('💾 Data Loading and Preprocessing...')
# Define paths
TRAIN_DIR = 'data/train_v2'
TEST_DIR = 'data/test_v2'

# Define the desired number of samples per ship count group
SAMPLES_PER_GROUP = 8000

# Define the batch size for training
BATCH_SIZE = 64

# Define the shape of the input images
IMAGE_SHAPE = (64, 64)

# Define the number of epochs for training
EPOCHS = 5

# Define the maximum number of training steps
MAX_TRAIN_STEPS = 200

# Data Exploration and Preprocessing
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

# Merge the 'segmentations' and 'ship_counts' DataFrames based on the 'image_id' column
data_merged = pd.merge(segmentations, ship_counts)

# Balance the data by sampling from each ship count group
data_merged_balanced = data_merged.groupby('ship_count').apply(
    lambda x: x.sample(SAMPLES_PER_GROUP) 
    if len(x) > SAMPLES_PER_GROUP 
    else x
).reset_index(drop=True)

# Split the balanced data into training and validation sets
train_df, valid_df = train_test_split(
    data_merged_balanced,
    test_size=0.01,
    stratify=data_merged_balanced['ship_count'],
)

print('🧬 Data Augmentation...')

# Define the augmentation parameters
generator_args = dict(
    rotation_range=20,         # Degree range for random rotations
    width_shift_range=0.1,     # Fraction of total width for random horizontal shifts
    height_shift_range=0.1,    # Fraction of total height for random vertical shifts
    zoom_range=0.05,           # Range for random zoom
    horizontal_flip=True,      # Randomly flip inputs horizontally
    vertical_flip=True,        # Randomly flip inputs vertically
    data_format='channels_last',  # Image data format ('channels_last' for (batch, height, width, channels))
)

# Create ImageDataGenerator instances for input images and labels separately
X_generator = ImageDataGenerator(**generator_args)  # For input images
y_generator = ImageDataGenerator(**generator_args)  # For labels (ground truth)

# Generate a validation image data generator and get a batch of validation data
X_val, y_val = next(get_image_generator(valid_df, valid_df.shape[0], IMAGE_SHAPE, TRAIN_DIR))

print('🤖 Model Building...')

# Model Definition and Compilation
# Create the U-Net model
model = unet_model()

# Compile the model
model.compile(
    optimizer='adam', 
    loss=dice_loss, 
    metrics=[dice_coefficient],
)

print('🚀 Model Training...')

# Final Training Steps
# Final augmented data being used in training
aug_gen = augment(get_image_generator(train_df, BATCH_SIZE, IMAGE_SHAPE, TRAIN_DIR), X_generator, y_generator)

# Best model weights
weight_path="{}_weights.best.hdf5".format('unet_model')

# Monitor validation dice coeff and save the best model weights
checkpoint = ModelCheckpoint(
    weight_path, 
    monitor='val_dice_coef', 
    verbose=2, 
    save_best_only=True, 
    mode='max', 
    save_weights_only=True,
)

# Reduce Learning Rate on Plateau
reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_dice_coefficient', 
    factor=0.5, 
    patience=3, 
    verbose=2, 
    mode='max', 
    min_delta=0.0001, 
    cooldown=2, 
    min_lr=1e-6,
)

# Stop training once there is no improvement seen in the model
early = EarlyStopping(
    monitor="val_dice_coefficient", 
    mode="max", 
    patience=15,
)

# Callbacks ready
callbacks_list = [checkpoint, early, reduceLROnPlat]

# Finalizing steps per epoch
step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)

# Save loss history while training
loss_history = model.fit(
    aug_gen,
    steps_per_epoch=step_count, 
    epochs=EPOCHS, 
    validation_data=(X_val, y_val),
    callbacks=callbacks_list,
    verbose=1,
)

print('👨🏻‍🎨 Loss Visualization...')

# Visualization
# Plot loss
plot_loss(loss_history)

# Plot Dice Coefficient
plot_dice_coefficient(loss_history)

print('💽 Saving Model...')

# Save the weights to load it later for test data 
model.save('models/unet_model.h5')

print('✅ Done!')
