import tensorflow as tf  # Import TensorFlow library

def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient between two binary tensors.

    Parameters:
    - y_true: True binary labels.
    - y_pred: Predicted binary labels.

    Returns:
    float: Dice coefficient.

    The Dice coefficient is a measure of similarity between two sets.
    It is defined as 2 * (intersection of sets) / (sum of sets).

    Note:
    The small epsilon (1e-5) is added to avoid division by zero.
    """

    # Compute the element-wise product of true and predicted binary labels
    intersection = tf.reduce_sum(y_true * y_pred)

    # Compute the sum of true and predicted binary labels separately
    sum_true = tf.reduce_sum(y_true)
    sum_pred = tf.reduce_sum(y_pred)

    # Calculate the union of sets
    union = sum_true + sum_pred

    # Compute the Dice coefficient using the formula, add epsilon to avoid division by zero
    dice_coefficient = (2.0 * intersection + 1e-5) / (union + 1e-5)

    # Return the computed Dice coefficient
    return dice_coefficient


def dice_loss(y_true, y_pred):
    """
    Compute the Dice loss between two binary tensors.

    Parameters:
    - y_true: True binary labels.
    - y_pred: Predicted binary labels.

    Returns:
    float: Dice loss.

    The Dice loss is defined as 1 - Dice coefficient, where the Dice coefficient is
    a measure of similarity between two sets (2 * intersection / (sum of sets)).

    Note:
    The small epsilon (1e-5) is added to avoid division by zero.
    """
    
    # Set a small value for smoothing to avoid division by zero
    smooth = 1.

    # Flatten the true and predicted binary labels
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    # Calculate the intersection of sets
    intersection = y_true_f * y_pred_f

    # Calculate the Dice coefficient using the formula, add smoothing
    score = (2. * tf.keras.backend.sum(intersection) + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    # Compute the Dice loss as 1 minus the Dice coefficient
    dice_loss = 1. - score

    # Return the computed Dice loss
    return dice_loss