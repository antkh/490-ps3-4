
Overview:
This project implements a simple feed-forward neural network using TensorFlow/Keras, trained on the *Iris dataset*. The goal is to classify iris flowers into three species based on four features: sepal length, sepal width, petal length, and petal width. This project serves as an introduction to neural networks and demonstrates how to use GitHub for machine learning projects.

Dataset:
The Iris dataset is a well-known dataset in machine learning, often used for testing classification algorithms. It includes:
- *150 samples*: Each sample represents an iris flower.
- *4 features* per sample: sepal length, sepal width, petal length, petal width.
- *3 classes*: Each sample belongs to one of three species (setosa, versicolor, virginica).


Model:
The neural network used in this project is a simple feed-forward architecture with the following structure:
1. Input: Accepts the four features of the dataset.
2. Two Hidden Layers:
   - 1: 64 neurons, ReLU activation.
   - 2: 32 neurons, ReLU activation.
3. Output: 3 neurons (one for each class), softmax activation to output class probabilities.


Structure:
- neural_network.py: Contains the NeuralNetwork class, which encapsulates the neural network model. The class includes functions to build, compile, train, and evaluate the model:
  - __init__(input_shape, num_classes): Initializes the model architecture.
  - compile_model(): Compiles the model with the optimizer, loss function, and metrics.
  - train(X_train, y_train, epochs, batch_size): Trains the model on the training dataset.
  - evaluate(X_test, y_test): Evaluates the model on the test dataset.
- example.ipynb: A Jupyter notebook that demonstrates how to use the NeuralNetwork class with the Iris dataset. This notebook includes:
  - Loading and preprocessing the dataset.
  - Initializing, compiling, and training the model.
  - Evaluating the model's performance.


Usage:
- You can use neural_network.py's NeuralNetwork class with its functions
- You can see the example usage in examply.ipynb (run the two blocks) 
