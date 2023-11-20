from tensorflow.keras import layers, models

def unet_model(input_shape=(64, 64, 3)):
    """
    Build a U-Net model for image segmentation.

    Parameters:
    - input_shape: Tuple, shape of the input images (default is (64, 64, 3)).

    Returns:
    models.Model: U-Net model.
    """
    
    # Input layer
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(4, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(4, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(8, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(8, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(16, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom
    conv4 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(conv4)
    up5 = layers.concatenate([up5, conv3])
    conv5 = layers.Conv2D(16, 3, activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv5)

    up6 = layers.Conv2DTranspose(8, 2, strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv2])
    conv6 = layers.Conv2D(8, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(8, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(4, 2, strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv1])
    conv7 = layers.Conv2D(4, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(4, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)

    # Define the model with input and output layers
    model = models.Model(inputs=inputs, outputs=outputs)

    return model
