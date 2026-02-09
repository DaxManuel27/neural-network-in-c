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
}Network;

Network * network_create(int * layer_sizes, int num_layers);
void network_free(Network * net);

Matrix * network_forward(Network * net, Matrix * input);
void network_print_info(Network * net);

#endif