#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"

typedef struct {
    Matrix * weights;
    Matrix * bias;
}Layer;

typedef struct {
    Layer * layers;
    int num_layers;
    int * layer_sizes;
    int num_classes;
    Matrix ** activations;
    int activations_allocated;
}Network;

typedef struct{
    Matrix * dweights;
    Matrix * dbias;
}LayerGradients;

typedef struct{
    LayerGradients * gradients;
    int num_layers;
}NetworkGradients;

typedef struct {
    int * layer_sizes;
    int num_layers;
    float learning_rate;
}NetworkConfig;

Network * network_create_from_config(NetworkConfig * config);

Network * network_create(int * layer_sizes, int num_layers);
void network_free(Network * net);

void network_forward(Network * net, Matrix * input);
void network_print_info(Network * net);


NetworkGradients * network_backward(Network * net, Matrix * input, Matrix * output_gradient);
void network_update_weights(Network * net, NetworkGradients * grads, float learning_rate);
void gradients_free(NetworkGradients * grads);
Matrix * output_gradient(Matrix * predictions, int true_label);

#endif