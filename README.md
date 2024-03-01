# Convolutional Neural Network for Image Classification

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) designed for image classification tasks.

## Model Architecture

The neural network architecture is defined in the `Net` class within the provided code. Below is a breakdown of the layers used:

- **Convolutional Layers:**
  - `conv1`: Applies a 2D convolution with 1 input channel, 16 output channels, and a kernel size of 3x3.
  - `conv2`: Another 2D convolution layer with 16 input channels, 16 output channels, and a kernel size of 3x3.
  - `conv3`: A 2D convolutional layer with 16 input channels, 32 output channels, and a kernel size of 3x3.
  - `conv4`: The final 2D convolutional layer with 32 input channels, 32 output channels, and a kernel size of 3x3.

- **Batch Normalization Layers:**
  - Batch normalization layers (`bn1`, `bn2`, `bn3`, `bn4`) are applied after each convolutional layer to stabilize and speed up training.

- **Pooling Layers:**
  - `pool1`: Max pooling layer with a kernel size of 2x2 and a stride of 2.
  - `pool2`: Another max pooling layer with the same kernel size and stride.

- **Dropout Layer:**
  - `dropout`: Applies 2D dropout regularization with a dropout probability of 0.5.

- **Fully Connected Layer:**
  - `fc1`: Fully connected layer with 32 * 3 * 3 input features (output from the last convolutional layer) and 10 output features (corresponding to the classes).

## Forward Pass
The forward pass through the network is defined in the `forward` method of the `Net` class. Here's a step-by-step explanation:
1. Input `x` undergoes a series of convolutional, batch normalization, and ReLU activation operations.
2. Max pooling is applied after the first and second convolutional blocks (`pool1`, `pool2`).
3. After the last convolutional block, ReLU activation and batch normalization are applied.
4. The output is flattened and passed through a dropout layer.
5. Finally, the output is fed into a fully connected layer (`fc1`) and passed through a softmax activation function to obtain class probabilities. The logarithm of the softmax probabilities is returned.
