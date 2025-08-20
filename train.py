import numpy as np
import gzip

class NeuralNetwork:
    """
    Feedforward Neural Network Implementation with Backpropagation
    Keywords: Deep Learning, Neural Networks, Multi-Layer Perceptron (MLP), 
    Feedforward Networks, Weight Initialization, Xavier/Glorot Initialization
    """
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.gradientsWeights = []
        self.gradientsBiases = []
        self.iterations = 0

        # Weight Initialization using Xavier/Glorot Method
        # Keywords: Weight Initialization, Xavier Initialization, Glorot Initialization
        # np.random.seed(0)
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden Layers Network - Multi-Layer Architecture
        # Keywords: Hidden Layers, Deep Neural Networks, Layer-wise Training
        for i in range(len(hidden_layers)-1):
            # np.random.seed(0)
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))
        
        # Output Layer - Final Classification Layer
        # Keywords: Output Layer, Classification Layer, Softmax Layer
        # np.random.seed(0)
        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers)-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
        """
        Forward Propagation - Neural Network Inference
        Keywords: Forward Propagation, Neural Network Inference, Matrix Operations,
        Linear Transformations, Activation Functions, ReLU, Softmax, Numerical Stability
        """
        self.outputs = [inputs]
        self.outputsTesting = ["inputs"]

        for i in range(len(self.weights)):
            # Linear Transformation: Matrix Multiplication + Bias Addition
            # Keywords: Linear Layer, Dense Layer, Matrix Operations, Bias Terms
            self.outputs.append(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i])
            self.outputsTesting.append("dense")

            # Activation Functions: ReLU for Hidden Layers, Softmax for Output
            # Keywords: Activation Functions, ReLU (Rectified Linear Unit), Softmax,
            # Non-linear Transformations, Numerical Stability, Gradient Flow
            if i == len(self.weights)-1:
                # Softmax with Numerical Stability (Prevents Overflow/Underflow)
                # Keywords: Softmax, Numerical Stability, Exponential Function, Normalization
                finalOutput = np.exp(self.outputs[-1] - np.max(self.outputs[-1], axis=1, keepdims=True))
                finalOutput = finalOutput / np.sum(finalOutput, axis=1, keepdims=True)
                self.outputs.append(finalOutput)
                self.outputsTesting.append("softmax")
            else:
                # ReLU Activation: max(0, x) - Introduces Non-linearity
                # Keywords: ReLU, Non-linearity, Gradient Flow, Vanishing Gradient Problem
                self.outputs.append(np.maximum(0, self.outputs[-1]))
                self.outputsTesting.append("relu")
        
        return self.outputs[-1]
    
    def backwards(self, y_true):
        """
        Backpropagation Algorithm - Gradient Computation
        Keywords: Backpropagation, Gradient Descent, Chain Rule, Automatic Differentiation,
        Loss Function Gradients, Weight Gradients, Bias Gradients, Computational Graphs
        """
        # Softmax + Categorical Cross-Entropy Loss Gradient Computation
        # Keywords: Softmax Gradient, Cross-Entropy Loss, Loss Function Derivatives,
        # Gradient Computation, Chain Rule Application

        # Number of samples for batch processing
        samples = len(self.outputs[-1])

        # Handle One-Hot Encoded Labels - Convert to Sparse Labels
        # Keywords: One-Hot Encoding, Sparse Labels, Label Processing, Data Preprocessing
        if len(y_true.shape) == 2:
            print("Changing to Discrete Values")
            y_true = np.argmax(y_true, axis=1)

        # Gradient of Softmax + Cross-Entropy Loss (Combined Derivative)
        # Keywords: Combined Gradient, Softmax-CrossEntropy Gradient, Loss Derivatives
        dSoftMaxCrossEntropy = self.outputs[-1].copy()
        # Apply Chain Rule: Subtract 1 from correct class predictions
        dSoftMaxCrossEntropy[range(samples), y_true] -= 1
        # Normalize gradient by batch size
        dSoftMaxCrossEntropy = dSoftMaxCrossEntropy / samples

        # Backpropagate through Output Layer
        # Keywords: Gradient Backpropagation, Weight Gradients, Bias Gradients, Input Gradients
        dInputs = np.dot(dSoftMaxCrossEntropy.copy(), self.weights[-1].T)

        # Compute Weight and Bias Gradients for Output Layer
        # Keywords: Weight Gradients, Bias Gradients, Matrix Operations, Gradient Accumulation
        dWeights = np.dot(self.outputs[-3].T, dSoftMaxCrossEntropy.copy())
        dBiases = np.sum(dSoftMaxCrossEntropy.copy(), axis=0, keepdims=True)
        self.gradientsWeights = [dWeights] + self.gradientsWeights
        self.gradientsBiases = [dBiases] + self.gradientsBiases

        # Backpropagate through Hidden Layers
        # Keywords: Hidden Layer Gradients, ReLU Gradients, Layer-wise Backpropagation
        i = -3
        j = -1
        for _ in range(len(self.hidden_layers)):
            i -= 1
            j -= 1
            
            # ReLU Gradient: Zero out gradients for negative inputs
            # Keywords: ReLU Gradient, Gradient Flow, Vanishing Gradient Prevention
            dInputsReLU = dInputs.copy()
            dInputsReLU[self.outputs[i] <= 0] = 0
            
            i -= 1
            # Continue backpropagation through linear layers
            # Keywords: Linear Layer Gradients, Matrix Transpose Operations
            dInputs = np.dot(dInputsReLU, self.weights[j].T)
            dWeights = np.dot(self.outputs[i].T, dInputsReLU)
            dBiases = np.sum(dInputsReLU, axis=0, keepdims=True)
            self.gradientsWeights = [dWeights] + self.gradientsWeights
            self.gradientsBiases = [dBiases] + self.gradientsBiases
    
    def updateParams(self, lr=0.05, decay=1e-7):
        """
        Parameter Update using Gradient Descent with Learning Rate Decay
        Keywords: Gradient Descent, Parameter Updates, Learning Rate Scheduling,
        Weight Updates, Bias Updates, Optimization Algorithms, Learning Rate Decay
        """
        # Learning Rate Decay: Reduces learning rate over time
        # Keywords: Learning Rate Decay, Adaptive Learning Rate, Convergence Optimization
        lr = lr * (1. / (1. + decay * self.iterations))

        # Update Weights using Gradient Descent
        # Keywords: Weight Updates, Gradient Descent, Parameter Optimization
        for i in range(len(self.weights)-1):
            if i != len(self.weights)-1:
                # Shape validation for gradient compatibility
                assert self.weights[i].shape == self.gradientsWeights[i].shape
                # Gradient Descent: w = w - learning_rate * gradient
                self.weights[i] += -lr*self.gradientsWeights[i]
        
        # Update Biases using Gradient Descent
        # Keywords: Bias Updates, Gradient Descent, Parameter Optimization
        for i in range(len(self.biases)-1):
            if i != len(self.biases)-1:
                # Shape validation for gradient compatibility
                assert self.biases[i].shape == self.gradientsBiases[i].shape
                # Gradient Descent: b = b - learning_rate * gradient
                self.biases[i] += -lr*self.gradientsBiases[i]
        
        # Increment iteration counter for learning rate decay
        self.iterations += 1

# Categorical Cross-Entropy Loss Function Implementation
def LossCategoricalCrossEntropy(yPred, yTrue):
    """
    Categorical Cross-Entropy Loss Function for Multi-Class Classification
    Keywords: Cross-Entropy Loss, Loss Functions, Multi-Class Classification,
    Information Theory, KL Divergence, Log Loss, Negative Log Likelihood
    """
    # Numerical Stability: Prevent log(0) by clipping predictions
    # Keywords: Numerical Stability, Log Function, Overflow Prevention, Underflow Prevention
    yPred = np.clip(yPred, 1e-10, 1 - 1e-10)

    # Compute Cross-Entropy Loss: -sum(y_true * log(y_pred))
    # Keywords: Cross-Entropy, Log Loss, Information Theory, Loss Computation
    loss = -np.sum(yTrue * np.log(yPred), axis=1)

    # Average Loss across Batch: Mean of individual sample losses
    # Keywords: Batch Processing, Loss Averaging, Training Metrics
    average_loss = np.mean(loss)

    return average_loss

def sparse_to_one_hot(sparse_labels, num_classes):
    """
    Convert Sparse Labels to One-Hot Encoded Format
    Keywords: One-Hot Encoding, Label Encoding, Data Preprocessing, 
    Categorical Variables, Multi-Class Classification
    """
    # Initialize zero matrix for one-hot encoding
    one_hot_encoded = np.zeros((len(sparse_labels), num_classes))
    # Set 1 at the position corresponding to the class label
    one_hot_encoded[np.arange(len(sparse_labels)), sparse_labels] = 1
    return one_hot_encoded

def extract_images(filename):
    """
    Extract MNIST Image Data from Binary Files
    Keywords: Data Loading, Binary File Processing, MNIST Dataset, 
    Image Processing, Data Preprocessing, File I/O Operations
    """
    with gzip.open(filename, 'rb') as f:
        # Parse MNIST binary format header
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32, count=4).byteswap()
        # Load image data and reshape to (num_samples, height, width)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def extract_labels(filename):
    """
    Extract MNIST Label Data from Binary Files
    Keywords: Data Loading, Binary File Processing, MNIST Dataset,
    Label Processing, Data Preprocessing, File I/O Operations
    """
    with gzip.open(filename, 'rb') as f:
        # Parse MNIST binary format header
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32, count=2).byteswap()
        # Load label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

myNeuralNet = NeuralNetwork(hidden_layers=[128, 128])

train_images = extract_images("./MNISTdata/train-images-idx3-ubyte.gz")
train_labels = extract_labels("./MNISTdata/train-labels-idx1-ubyte.gz")
test_images = extract_images("./MNISTdata/t10k-images-idx3-ubyte.gz")
test_labels = extract_labels("./MNISTdata/t10k-labels-idx1-ubyte.gz")

# MNIST Training Pipeline - Neural Network Training Loop
# Keywords: Training Loop, Epochs, Batch Processing, Stochastic Gradient Descent,
# Model Training, Supervised Learning, Deep Learning Training

data = train_images
dataLabels = train_labels

# Data Normalization: Scale pixel values to [-1, 1] range
# Keywords: Data Normalization, Feature Scaling, Data Preprocessing, 
# Min-Max Scaling, Input Standardization
data = (data.astype(np.float32)-127.5)/127.5

# Reshape images to 1D vectors for neural network input
# Keywords: Data Reshaping, Feature Vectorization, Input Preparation
data = data.reshape(60000, 784)

# Training metrics tracking
accuracies = []
losses = []

# Batch size for mini-batch gradient descent
# Keywords: Batch Size, Mini-Batch Gradient Descent, Stochastic Optimization
BATCH_SIZE = 32

# Main Training Loop: Epoch-based Training
# Keywords: Training Loop, Epochs, Iterative Training, Convergence
for epoch in range(1, 5):
    print(f'epoch: {epoch}')
    train_steps = len(data) // BATCH_SIZE

    # Mini-batch Training Loop
    # Keywords: Mini-Batch Processing, Stochastic Gradient Descent, Batch Training
    for step in range(train_steps):
        # Create mini-batch for current training step
        # Keywords: Batch Creation, Data Sampling, Mini-Batch Formation
        batch_X = data[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        batch_y = dataLabels[step*BATCH_SIZE:(step+1)*BATCH_SIZE]

        X = batch_X
        y = batch_y

        # Forward Pass: Neural Network Inference
        # Keywords: Forward Propagation, Neural Network Inference, Prediction
        output = myNeuralNet.forward(X)

        # Training Metrics Evaluation (every 100 steps)
        # Keywords: Training Metrics, Accuracy Evaluation, Loss Monitoring, Model Evaluation
        if step % 100 == 0:
            # Compute predictions and accuracy
            # Keywords: Prediction Generation, Accuracy Computation, Classification Metrics
            predictions = np.argmax(output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(predictions==y)

            # Compute loss for monitoring
            # Keywords: Loss Computation, Training Loss, Model Performance Metrics
            loss = LossCategoricalCrossEntropy(output, sparse_to_one_hot(y, 10))
            accuracies.append(accuracy)
            losses.append(loss)

            print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f}')
        
        # Backward Pass: Gradient Computation
        # Keywords: Backpropagation, Gradient Computation, Error Backpropagation
        myNeuralNet.backwards(y)

        # Parameter Updates: Gradient Descent Optimization
        # Keywords: Parameter Updates, Gradient Descent, Optimization Step
        myNeuralNet.updateParams(lr=0.5, decay=1e-6)

dataTest = test_images
dataTestLabels = test_labels

# Normalize
dataTest = (dataTest.astype(np.float32)-127.5)/127.5
print(dataTest.shape)
dataTest = dataTest.reshape(10000, 784)

X = dataTest
y = dataTestLabels

output = myNeuralNet.forward(dataTest)

predictions = np.argmax(output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print(accuracy)

# Save the trained model
print("Saving trained model...")
for i, weight in enumerate(myNeuralNet.weights):
    np.save(f"trained_weights_{i}.npy", weight)
for i, bias in enumerate(myNeuralNet.biases):
    np.save(f"trained_biases_{i}.npy", bias)
print("Model saved!")
