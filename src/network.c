#include "network.h"

Network* network_create(int *layer_sizes, int num_layers) {
    Network *net = malloc(sizeof(Network));
    net->num_layers = num_layers - 1;  // Number of weight matrices (connections between layers)
    net->layer_sizes = malloc(num_layers * sizeof(int));
    
    for (int i = 0; i < num_layers; i++) {
        net->layer_sizes[i] = layer_sizes[i];
    }

    net->layers = malloc(net->num_layers * sizeof(Layer));
    
    // TODO: Create weight and bias matrices for each layer
    // For layer i: weights should be (layer_sizes[i] x layer_sizes[i+1])
    //             bias should be (1 x layer_sizes[i+1])
    for(int i = 0; i < net->num_layers; i++){
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];
        net->layers[i].weights = matrix_create(input_size, output_size);
        net->layers[i].bias = matrix_zeros(1, output_size);
    }
    printf("Network created with %d layers\n", net->num_layers);
    for (int i = 0; i < net->num_layers; i++) {
        printf("  Layer %d: %d -> %d\n", i, 
               net->layers[i].weights->rows, 
               net->layers[i].weights->cols);
    }
    
    return net;
}

void network_free(Network * net){
    if(!net){
        return;
    }

    for(int i = 0; i < net->num_layers; i++){
        matrix_free(net->layers[i].weights);
        matrix_free(net->layers[i].bias);
    }
    free(net->layers);
    free(net->layer_sizes);
    free(net);
}
Matrix* network_forward(Network *net, Matrix *input) {
    if (!net || !input) {
        printf("Error: NULL network or input\n");
        return NULL;
    }
    
    Matrix *current = input;
    
    for (int i = 0; i < net->num_layers; i++) {
        // Step 1: Multiply by weights
        Matrix *z = matrix_multiply(current, net->layers[i].weights);
        
        // Step 2: Add bias
        matrix_add(z, net->layers[i].bias);
        
        // Step 3: Apply activation
        if (i < net->num_layers - 1) {
            // Hidden layer: use ReLU
            matrix_relu(z);
        } else {
            // Output layer: use softmax
            matrix_softmax(z);
        }
        
        // Step 4: Free old current (only if it's not the input)
        if (current != input) {
            matrix_free(current);
        }
        
        // Step 5: Move to next iteration
        current = z;
    }
    
    return current;
}