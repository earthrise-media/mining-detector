# A collection of models explored, Aug / Sept 2025

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50


# 2024 v3 CNN ~ 100k params for 48px patch input.

def CNN2024(input_shape=(48,48,13)):
    """Construct the CNN used in 2024."""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.MaxPooling2D(pool_size=(2)),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.MaxPooling2D(pool_size=(2)),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.Conv2D(32, kernel_size=(3), padding='same', activation="relu"),
        layers.MaxPooling2D(pool_size=(3)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')])
    
    return model

# 800k param CNN

def CNN800k(input_shape=(48,48,13)):
    """Construct a CNN with ~800k parameters, given 48px patch input."""

    model = keras.Sequential([
        keras.Input(shape=input_shape),
    
        # Block 1
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Block 2
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        # Block 3
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),

        # Fully connected layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
    
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# ResNet-18, 11M params

def conv_block(x, filters, kernel_size=3, stride=1, use_bias=False, weight_decay=1e-4):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def residual_block(x, filters, stride=1, downsample=None, weight_decay=1e-4):
    shortcut = x

    # First conv
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second conv
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    # Downsample shortcut if needed
    if downsample is not None:
        shortcut = downsample

    # Add + ReLU
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def make_layer(x, filters, blocks, stride=1, weight_decay=1e-4):
    # Downsample if stride != 1 or filter mismatch
    downsample = None
    input_filters = x.shape[-1]
    if stride != 1 or input_filters != filters:
        downsample = layers.Conv2D(
            filters, 1, strides=stride, padding='same', use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay))(x)
        downsample = layers.BatchNormalization()(downsample)

    x = residual_block(x, filters, stride=stride, downsample=downsample,
                       weight_decay=weight_decay)
    for _ in range(1, blocks):
        x = residual_block(x, filters, stride=1, weight_decay=weight_decay)
    return x

def ResNet18(input_shape=(48,48,13), num_classes=1, weight_decay=1e-4):
    """Construct a ResNet-18 model."""
    inputs = layers.Input(shape=input_shape)

    # First conv adapted for 13 channels
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual layers
    x = make_layer(x, 64, blocks=2, stride=1, weight_decay=weight_decay)
    x = make_layer(x, 128, blocks=2, stride=2, weight_decay=weight_decay)
    x = make_layer(x, 256, blocks=2, stride=2, weight_decay=weight_decay)
    x = make_layer(x, 512, blocks=2, stride=2, weight_decay=weight_decay)

    # Global pooling + dense
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

# WIP ResNet 50. Adjusted for 13 input bands but the pretrained
# model still expects (224,224) rasters. 

def ResNet50(input_shape=(48,48,13), num_classes=1, weight_decay=1e-4):
    """Construct a ResNet-50 pretrained on ImageNet, for input 13 bands."""
    inputs = layers.Input(shape=input_shape)

    # --- First conv replacement ---
    # Original ResNet50 expects 3 channels; we use 13 instead
    x = layers.Conv2D(
        64, 7, strides=2, padding='same', use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # --- Load ResNet50 without the top layers ---
    # Use include_top=False, no input layer needed since we manually handle it
    base = ResNet50(
        include_top=False,
        weights='imagenet',   # pretrained on ImageNet
        input_tensor=x,
        pooling=None
    )

    # --- Global Pooling + Classification Head ---
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model



def MLP(
    input_dim: int = 384,
    hidden_layers: tuple = (64, 16),
    activation: str = "relu",
    output_activation: str = "sigmoid"
) -> tf.keras.Model:
    """
    Build a fully connected MLP for binary classification.

    Args:
        input_dim (int): Dimension of the input features (default 384).
        hidden_layers (tuple): Sizes of hidden layers, e.g. (64, 16).
        activation (str): Activation for hidden layers (default "relu").
        output_activation (str): Activation for output layer (default "sigmoid").

    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))

    # Output layer: single neuron for binary classification
    model.add(layers.Dense(1, activation=output_activation))

    return model
