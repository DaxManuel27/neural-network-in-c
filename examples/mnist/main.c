#include "mnist.h"
#include "../src/matrix.h"
#include "../src/network.h"
#include "../src/loss.h"
#include <stdio.h>


int main() {
    printf("=== MNIST Neural Network ===\n\n");
    
    // Step 1: Load MNIST data
    printf("Loading MNIST dataset...\n");
    MNISTDataset *train = load_mnist_images("train-images-idx3-ubyte");
    if (!train) {
        printf("Failed to load images\n");
        return 1;
    }
    load_mnist_labels("train-labels-idx1-ubyte", train);
    
    // Step 2: Create network
    printf("\nCreating network...\n");
    int layer_sizes[] = {784, 128, 64, 10};
    Network *net = network_create(layer_sizes, 4);
    
    // Step 3: Test forward pass on first image
    printf("\nTesting forward pass...\n");
    Matrix *image = matrix_zeros(1, 784);
    for (int j = 0; j < 784; j++) {
        image->data[j] = train->images[0].pixels[j];
    }
    
    network_forward(net, image);
    Matrix *output = net->activations[net->num_layers - 1];
    
    printf("Input: Image of digit %d\n", train->images[0].label);
    printf("Output predictions:\n");
    for (int i = 0; i < 10; i++) {
        printf("  Digit %d: %.4f\n", i, output->data[i]);
    }
    
    //training loop 
    float learning_rate = 0.01f;
    int num_epochs = 100;

    for(int epoch = 0; epoch < num_epochs; epoch++){
        float total_loss = 0.0f;
        int correct = 0;

        for (int img_idx = 0; img_idx < train->count; img_idx++) {
            // Step 1: Copy image to matrix
            for (int j = 0; j < 784; j++) {
                image->data[j] = train->images[img_idx].pixels[j];
            }
            
            // Step 2: Forward pass
            network_forward(net, image);
            Matrix *output = net->activations[net->num_layers - 1];
            
            // Step 3: Calculate loss
            float loss = calculate_loss(output, train->images[img_idx].label);
            total_loss += loss;
            
            // Step 4: Calculate output gradient
            Matrix *output_grad = output_gradient(output, train->images[img_idx].label);
            
            // Step 5: Backpropagate
            NetworkGradients *grads = network_backward(net, image, output_grad);
            
            // Step 6: Update weights
            network_update_weights(net, grads, learning_rate);
            
            // Step 7: Track accuracy
            // Find the predicted digit (highest probability)
            int predicted_digit = 0;
            float max_prob = output->data[0];
            for (int d = 1; d < 10; d++) {
                if (output->data[d] > max_prob) {
                    max_prob = output->data[d];
                    predicted_digit = d;
                }
            }
            
            // Check if prediction is correct
            if (predicted_digit == train->images[img_idx].label) {
                correct++;
            }
            
            // Cleanup for this iteration
            matrix_free(output_grad);
            gradients_free(grads);
            
            // Progress update
            if ((img_idx + 1) % 10000 == 0) {
                printf("  Processed %d images\n", img_idx + 1);
            }
        }
        printf("  Loss: %.4f, Accuracy: %.2f%%\n", 
            total_loss / train->count, 
            (100.0f * correct) / train->count);
            
        }
        // Load test data
    printf("\nLoading test dataset...\n");
    MNISTDataset *test = load_mnist_images("t10k-images-idx3-ubyte");
    if (!test) {
        printf("Failed to load test images\n");
        return 1;
    }
    load_mnist_labels("t10k-labels-idx1-ubyte", test);
    // Step 5: Test on test set
    printf("\n=== Testing ===\n");
    {
        float test_loss = 0.0f;
        int test_correct = 0;
        
        for (int img_idx = 0; img_idx < test->count; img_idx++) {
            // Copy test image
            for (int j = 0; j < 784; j++) {
                image->data[j] = test->images[img_idx].pixels[j];
            }
            
            // Forward pass only (no backprop)
            network_forward(net, image);
            Matrix *output = net->activations[net->num_layers - 1];
            
            // Calculate loss
            float loss = calculate_loss(output, test->images[img_idx].label);
            test_loss += loss;
            
            // Find predicted digit
            int predicted_digit = 0;
            float max_prob = output->data[0];
            for (int d = 1; d < 10; d++) {
                if (output->data[d] > max_prob) {
                    max_prob = output->data[d];
                    predicted_digit = d;
                }
            }
            
            // Check if correct
            if (predicted_digit == test->images[img_idx].label) {
                test_correct++;
            }
            
            if ((img_idx + 1) % 2500 == 0) {
                printf("  Tested %d images\n", img_idx + 1);
            }
        }
        
        printf("Test Loss: %.4f, Test Accuracy: %.2f%%\n", 
               test_loss / test->count, 
               (100.0f * test_correct) / test->count);
    }

    // Step 6: Cleanup
    printf("\nCleaning up...\n");
    matrix_free(image);
    network_free(net);
    mnist_dataset_free(train);
    mnist_dataset_free(test);
    
    printf("Done!\n");
    return 0;

}