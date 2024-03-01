import streamlit as st

# Title
st.title("Fashion MNIST Classification with Convolutional Neural Networks")

# Description
st.markdown("""
             This Streamlit app demonstrates a simple Convolutional Neural Network (CNN) model for classifying 
             fashion images from the Fashion MNIST dataset. The model is trained on a subset of the Fashion MNIST 
             dataset and then evaluated on a separate test set. Additionally, the trained model is saved for future use.
             """)

# Display Source Code
if st.button("Show Source Code"):
    st.text("Source Code:")
    st.code("""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Define class labels
class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# Preprocess data
X_train, X_test = X_train[..., np.newaxis] / 255.0, X_test[..., np.newaxis] / 255.0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2020)

# Define model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=512, verbose=1, validation_data=(X_val, y_val))

# Evaluate model
model.evaluate(X_test, y_test)

# Save model
model.save('fashion.h5')
            """)
