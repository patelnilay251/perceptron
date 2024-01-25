import numpy as np
import matplotlib.pyplot as plt


# Function to generate grayscale images of handwritten labeled numbers
def generate_images(size, num_images):
    images = []
    labels = []

    for label in range(10):
        for _ in range(num_images):
            image = np.random.rand(size, size)
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


# Function to initialize parameters (transmission matrix and bias)
def initialize_parameters(input_size, output_size):
    transmission_matrix = np.random.rand(output_size, input_size)
    bias = np.random.rand(output_size, 1)

    return transmission_matrix, bias


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Loss function
def compute_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)


# Forward propagation
def forward_propagation(inputs, transmission_matrix, bias):
    linear_combination = np.dot(transmission_matrix, inputs) + bias
    output = sigmoid(linear_combination)
    return output


# Backward propagation
def backward_propagation(
    inputs, targets, output, transmission_matrix, bias, learning_rate
):
    error = targets - output
    gradient = output * (1 - output) * error
    transmission_matrix += learning_rate * np.outer(gradient, inputs)
    bias += learning_rate * gradient

    return transmission_matrix, bias


# Training the perceptron
def train_perceptron(images, labels, epochs, learning_rate):
    input_size = images.shape[1] * images.shape[2]
    output_size = 10
    transmission_matrix, bias = initialize_parameters(input_size, output_size)

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(images)):
            flattened_image = images[i].flatten().reshape((input_size, 1))
            target = np.zeros((output_size, 1))
            target[labels[i]] = 1

            output = forward_propagation(flattened_image, transmission_matrix, bias)
            total_loss += compute_loss(output, target)
            transmission_matrix, bias = backward_propagation(
                flattened_image,
                target,
                output,
                transmission_matrix,
                bias,
                learning_rate,
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(images)}")

    return transmission_matrix, bias


# Testing the perceptron
def test_perceptron(test_images, transmission_matrix, bias):
    predictions = []

    for i in range(len(test_images)):
        flattened_image = test_images[i].flatten().reshape((-1, 1))
        output = forward_propagation(flattened_image, transmission_matrix, bias)
        predicted_label = np.argmax(output)
        predictions.append(predicted_label)

    return predictions


# Generate and display training images
training_images, training_labels = generate_images(20, 10)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(training_images[i], cmap="gray")
    plt.title(f"Label: {training_labels[i]}")
    plt.axis("off")
plt.show()

# Train the perceptron
epochs = 100
learning_rate = 0.01
transmission_matrix, bias = train_perceptron(
    training_images, training_labels, epochs, learning_rate
)

# Optional: Visualize training loss

# Generate and display test images
test_images, _ = generate_images(20, 5)
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i], cmap="gray")
    plt.axis("off")
plt.show()

# Test the perceptron
test_predictions = test_perceptron(test_images, transmission_matrix, bias)
print("Test Predictions:", test_predictions)
