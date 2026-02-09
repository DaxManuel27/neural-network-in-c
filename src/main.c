#include <stdio.h>
#include "mnist.h"
#include "matrix.h"
#include "network.h"

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
    
    Matrix *output = network_forward(net, image);
    printf("Input: Image of digit %d\n", train->images[0].label);
    printf("Output predictions:\n");
    for (int i = 0; i < 10; i++) {
        printf("  Digit %d: %.4f\n", i, output->data[i]);
    }
    
    // Step 4: Cleanup
    printf("\nCleaning up...\n");
    matrix_free(image);
    matrix_free(output);
    network_free(net);
    mnist_dataset_free(train);
    
    printf("Done!\n");
    return 0;
}