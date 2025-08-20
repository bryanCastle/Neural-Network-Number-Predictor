# Neural Network from Scratch - MNIST Digit Classification

A comprehensive implementation of a feedforward neural network built entirely from scratch using NumPy, designed for MNIST digit classification. This project demonstrates fundamental deep learning concepts including backpropagation, gradient descent, and neural network architecture.

## 🎯 Project Overview

This repository contains a complete neural network implementation that achieves **~95% accuracy** on the MNIST test set. The implementation includes:

- **Pure NumPy Implementation**: No deep learning frameworks required
- **Complete Training Pipeline**: Forward/backward propagation with gradient descent
- **Interactive Drawing App**: Real-time digit prediction interface
- **Educational Focus**: Well-documented code with AI/ML keywords for learning

## 🏗️ Architecture

### Neural Network Structure
```
Input Layer: 784 neurons (28×28 flattened images)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 128 neurons (ReLU activation)  
Output Layer: 10 neurons (Softmax activation)
```

### Key Mathematical Concepts Implemented
- **Forward Propagation**: Matrix operations, linear transformations, activation functions
- **Backpropagation**: Chain rule, gradient computation, weight updates
- **Loss Function**: Categorical Cross-Entropy with numerical stability
- **Optimization**: Mini-batch gradient descent with learning rate decay

## 📁 Project Structure

```
NNBegginer/
├── train.py                           # Main training script
├── digit_drawing_app_preview.py       # Interactive drawing application
├── MNISTdata/                         # MNIST dataset files
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── modelWeights/                      # Pre-trained PyTorch weights (optional)
└── testImages/                        # Custom test images
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib pillow
```

### Training the Model
```bash
python train.py
```

### Running the Drawing App
```bash
python digit_drawing_app_preview.py
```

## 🔬 Technical Implementation

### Core Algorithms

#### 1. Forward Propagation
```python
# Linear transformation: Wx + b
output = np.dot(inputs, weights) + biases

# ReLU activation: max(0, x)
relu_output = np.maximum(0, linear_output)

# Softmax with numerical stability
softmax_output = np.exp(x - max(x)) / sum(exp(x - max(x)))
```

#### 2. Backpropagation
```python
# Gradient of softmax + cross-entropy
gradient = predictions - true_labels

# Weight gradients: ∂L/∂W = ∂L/∂z * ∂z/∂W
weight_gradients = np.dot(inputs.T, gradients)

# Backpropagate through layers using chain rule
```

#### 3. Parameter Updates
```python
# Gradient descent with learning rate decay
learning_rate = base_lr / (1 + decay * iteration)
weights = weights - learning_rate * weight_gradients
```

### Key Features

- **Numerical Stability**: Prevents overflow/underflow in softmax computation
- **Batch Processing**: Mini-batch gradient descent for efficient training
- **Learning Rate Decay**: Adaptive learning rate scheduling
- **Data Preprocessing**: MNIST normalization and reshaping
- **Model Persistence**: Save/load trained weights

## 🎨 Interactive Drawing Application

The `digit_drawing_app_preview.py` provides a real-time digit recognition interface:

- **28×28 Drawing Canvas**: Pixel-perfect MNIST format
- **Real-time Prediction**: Instant AI predictions with confidence scores
- **Probability Visualization**: Bar chart showing prediction probabilities
- **Custom Image Support**: Test with your own digit images

## 📊 Performance Metrics

### Training Results
- **Architecture**: 784 → [128, 128] → 10
- **Training Time**: ~2-3 minutes for 4 epochs
- **Final Accuracy**: ~95% on MNIST test set
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Mini-batch Gradient Descent

### Model Performance
```
Epoch 1: Accuracy ~85-90%, Loss ~0.5-1.0
Epoch 2: Accuracy ~90-93%, Loss ~0.3-0.6
Epoch 3: Accuracy ~92-95%, Loss ~0.2-0.4
Epoch 4: Accuracy ~94-96%, Loss ~0.1-0.3
```

## 🧠 AI/ML Concepts Demonstrated

### Deep Learning Fundamentals
- **Neural Networks**: Multi-layer perceptron architecture
- **Activation Functions**: ReLU, Softmax implementations
- **Loss Functions**: Cross-entropy loss with numerical stability
- **Optimization**: Gradient descent, learning rate scheduling

### Mathematical Concepts
- **Linear Algebra**: Matrix operations, dot products, transposes
- **Calculus**: Chain rule, partial derivatives, gradients
- **Statistics**: Probability distributions, softmax normalization
- **Numerical Methods**: Numerical stability, overflow prevention

### Machine Learning Pipeline
- **Data Preprocessing**: Normalization, reshaping, encoding
- **Model Training**: Forward/backward propagation, parameter updates
- **Model Evaluation**: Accuracy metrics, loss monitoring
- **Model Deployment**: Real-time inference, prediction serving

## 🎓 Educational Value

This project serves as an excellent learning resource for:

- **Deep Learning Fundamentals**: Understanding neural network internals
- **Mathematical Implementation**: Translating theory to code
- **AI/ML Keywords**: Industry-relevant terminology and concepts
- **Practical Skills**: Real-world neural network development

### Key Learning Outcomes
- Implement neural networks from first principles
- Understand backpropagation and gradient computation
- Master numerical stability in deep learning
- Build interactive AI applications
- Develop production-ready ML pipelines

## 🔧 Customization

### Architecture Modifications
```python
# Change network architecture
model = NeuralNetwork(
    input_size=784,
    hidden_layers=[256, 128, 64],  # Custom architecture
    output_size=10
)
```

### Training Parameters
```python
# Adjust training hyperparameters
myNeuralNet.updateParams(
    lr=0.1,        # Learning rate
    decay=1e-6     # Learning rate decay
)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Additional Optimizers**: Adam, RMSprop, momentum
- **Regularization**: Dropout, L1/L2 regularization
- **Advanced Architectures**: Convolutional layers, residual connections
- **Performance Optimization**: Vectorization, parallel processing
- **Additional Datasets**: Fashion-MNIST, CIFAR-10

## 📚 References

- **Neural Networks from Scratch**: https://nnfs.io/
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **Deep Learning Fundamentals**: Backpropagation, gradient descent
- **Numerical Methods**: Stability in neural network computations

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning and research.

---

**Keywords**: Deep Learning, Neural Networks, Backpropagation, Gradient Descent, MNIST Classification, NumPy Implementation, Machine Learning, AI Development, Computer Vision, Multi-Class Classification, Feedforward Networks, Activation Functions, Loss Functions, Optimization Algorithms, Data Preprocessing, Model Training, Real-time Inference, Interactive Applications
