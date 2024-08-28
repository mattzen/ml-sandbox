import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Define the autoencoder architecture
input_dim = 784  # 28x28 pixels
encoding_dim = 32  # Size of the encoded representation

input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Create and compile the autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

# Create separate encoder and decoder models
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-3](encoded_input)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(encoded_input, decoder_layer)

# Train the autoencoder
history = autoencoder.fit(x_train, x_train,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

# Plot the training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Encode and decode some digits from the test set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Display original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()

# Visualize the encoded space
plt.figure(figsize=(10, 10))
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, cmap='viridis')
plt.colorbar()
plt.title('2D Projection of Encoded Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# This example demonstrates how to use autoencoders for dimensionality reduction and 
# reconstruction of MNIST digits. Here's a breakdown of what the code does:

# We load and preprocess the MNIST dataset.
# We define an autoencoder architecture with an input layer, multiple dense layers for encoding and decoding, and an output layer.
# We compile and train the autoencoder using binary cross-entropy loss.
# We plot the training history to visualize how the loss decreases over time.
# We use the trained autoencoder to encode and decode test images.
# We visualize original and reconstructed images side by side.
# Finally, we create a scatter plot of the encoded space to see how different digits are clustered.

# This example showcases several practical applications of autoencoders:

# Dimensionality reduction: We compress 784-dimensional input (28x28 pixels) into a 32-dimensional encoded representation.
# Feature learning: The autoencoder learns to extract meaningful features from the input data.
# Noise reduction: By training the autoencoder to reconstruct input images, it learns to remove noise and capture essential features.
# Anomaly detection: You could use the reconstruction error to detect anomalies in new data.

# I've made several changes to address the error and improve the code:

# Created separate encoder and decoder models: Instead of trying to access autoencoder.encoder, we now create distinct encoder and decoder models.
# Used predict method: We now use encoder.predict() and decoder.predict() instead of trying to call the models directly.
# Added y_train and y_test: These were missing in the original data loading step, which caused an issue in the final visualization.
# Separated the plot commands: Each plt.show() is now called immediately after its respective plot is created, ensuring that all plots are displayed.

# This updated version should run without the previous error. Here's a brief explanation of what each part does:

# Data preparation: Load and normalize the MNIST dataset.
# Model definition: Create the autoencoder architecture.
# Model compilation: Set up the model with an optimizer and loss function.
# Training: Fit the model to the training data.
# Visualization of training history: Plot the loss over epochs.
# Image reconstruction: Use the encoder and decoder to process test images.
# Visualization of results: Display original and reconstructed images side by side.
# Encoded space visualization: Create a scatter plot of the encoded representations.