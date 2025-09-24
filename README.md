# Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 dataset**. The project includes a comparison with a standard **Artificial Neural Network (ANN)** model to highlight the effectiveness of CNNs for image-based tasks.

The code is written in Python and uses the **TensorFlow** and **Keras** libraries for building and training the models, and **Scikit-learn** for evaluation.

### Dataset

The project uses the **CIFAR-10 dataset**, a well-known dataset in computer vision. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

  * `airplane`
  * `automobile`
  * `bird`
  * `cat`
  * `deer`
  * `dog`
  * `frog`
  * `horse`
  * `ship`
  * `truck`

### Project Structure

The Jupyter notebook `Cnn.ipynb` contains the following sections:

#### Data Loading and Preprocessing

The CIFAR-10 dataset is loaded and split into training and testing sets. The image data is then normalized by dividing the pixel values by 255.0 to scale them between 0 and 1. The labels are reshaped to a 1D array for compatibility with the models. A function is included to display sample images with their corresponding labels.

<br>

-----

### ANN Model

An Artificial Neural Network (ANN) model is built and trained as a baseline for comparison. This model uses a `Flatten` layer to convert the 32x32x3 images into a 1D vector, followed by two dense hidden layers and a final dense layer with a `softmax` activation for classification.

  * **Layers:** `Flatten`, `Dense` (3000 neurons, `relu`), `Dense` (1000 neurons, `relu`), `Dense` (10 neurons, `softmax`)
  * **Optimizer:** `SGD`
  * **Loss Function:** `sparse_categorical_crossentropy`

The performance of the ANN model is evaluated using a **classification report**, which provides metrics like precision, recall, and F1-score.

<br>

-----

### CNN Model

A more advanced Convolutional Neural Network (CNN) model is implemented to improve classification accuracy. The CNN model is designed with multiple sets of **convolutional layers (`Conv2D`)** and **max-pooling layers (`MaxPooling2D`)** to automatically learn hierarchical features from the images.

  * **Architecture:**
      * Three blocks of `Conv2D` and `Conv2D` layers, followed by a `MaxPooling2D` layer.
      * Each block increases the number of filters (32, 64, 64) to capture more complex features.
      * A `Flatten` layer is used to transition from the convolutional blocks to the dense layers.
      * A dense hidden layer with 521 neurons and a final output layer with 10 neurons and `softmax` activation are used for classification.
  * **Optimizer:** `adam`
  * **Loss Function:** `sparse_categorical_crossentropy`
  * **Training:** The model is trained for 10 epochs.

The output of the notebook shows that the CNN model achieves a significantly higher accuracy on the training data compared to the ANN model, demonstrating its superiority for image classification tasks.

### Dependencies

  * `numpy`
  * `matplotlib`
  * `tensorflow`
  * `keras`
  * `scikit-learn`
