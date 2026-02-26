#include "network.h"

Network* network_create(int *layer_sizes, int num_layers) {
    Network *net = malloc(sizeof(Network));
    net->num_layers = num_layers - 1;
    net->layer_sizes = malloc(num_layers * sizeof(int));
    
    for (int i = 0; i < num_layers; i++) {
        net->layer_sizes[i] = layer_sizes[i];
    }
    net->layers = malloc(net->num_layers * sizeof(Layer));
    net->activations = malloc(net->num_layers * sizeof(Matrix *));
    net->activations_allocated = 0;
    net->num_classes = layer_sizes[num_layers - 1];
    for (int i = 0; i < net->num_layers; i++) {
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

void network_free(Network *net) {
    if (!net) return;
    for (int i = 0; i < net->activations_allocated; i++) {
        matrix_free(net->activations[i]);
    }
    free(net->activations);
    for (int i = 0; i < net->num_layers; i++) {
        matrix_free(net->layers[i].weights);
        matrix_free(net->layers[i].bias);
    }
    
    free(net->layers);
    free(net->layer_sizes);
    free(net);
}
void network_forward(Network *net, Matrix *input) {
    if (!net || !input) {
        printf("Error: NULL network or input\n");
        return;
    }
    

    for (int i = 0; i < net->activations_allocated; i++) {
        matrix_free(net->activations[i]);
    }
    net->activations_allocated = 0;
    
    Matrix *current = input;
    
    for (int i = 0; i < net->num_layers; i++) {
        Matrix *z = matrix_multiply(current, net->layers[i].weights);
        

        matrix_add(z, net->layers[i].bias);
        

        if (i < net->num_layers - 1) {

            matrix_relu(z);
        } else {

            matrix_softmax(z);
        }
        
        net->activations[i] = z;
        net->activations_allocated++;
        

        if (current != input) {
        }
        current = z;
    }
}


Matrix * output_gradient(Matrix * predictions, int true_label){
    if(!predictions || true_label < 0 || true_label >= 10){
        return NULL;
    }
    Matrix * grad = matrix_zeros(1,10);
    for(int i = 0; i < 10; i++){
        grad->data[i] = predictions->data[i];
    }
    grad->data[true_label] -= 1.0f;
    return grad;
}

NetworkGradients* network_backward(Network *net, Matrix *input, Matrix *output_grad) {
    if (!net || !input || !output_grad) {
        printf("Error: NULL arguments in network_backward\n");
        return NULL;
    }
    
    NetworkGradients *grads = malloc(sizeof(NetworkGradients));
    grads->num_layers = net->num_layers;
    grads->gradients = malloc(net->num_layers * sizeof(LayerGradients));
    
    Matrix *current_grad = output_grad;
    

    for (int i = net->num_layers - 1; i >= 0; i--) {

        Matrix *layer_input;
        if (i == 0) {
            layer_input = input;
        } else {
            layer_input = net->activations[i - 1];
        }
        
        // calc weight gradient
        Matrix *input_T = matrix_transpose(layer_input);
        Matrix *dweights = matrix_multiply(input_T, current_grad);
        matrix_free(input_T);
        
        // bias gradient
        Matrix *dbias = matrix_zeros(1, current_grad->cols);
        for (int j = 0; j < current_grad->cols; j++) {
            dbias->data[j] = current_grad->data[j];
        }
        
        //prev layer gradient
        Matrix *weights_T = matrix_transpose(net->layers[i].weights);
        Matrix *grad_prev = matrix_multiply(current_grad, weights_T);
        matrix_free(weights_T);
        
        // relu for hidden layers
        if (i > 0) {
            Matrix *relu_deriv = matrix_relu_derivative(net->activations[i - 1]);
            
            for (int j = 0; j < grad_prev->rows * grad_prev->cols; j++) {
                grad_prev->data[j] *= relu_deriv->data[j];
            }
            
            matrix_free(relu_deriv);
        }

        grads->gradients[i].dweights = dweights;
        grads->gradients[i].dbias = dbias;
        if (i < net->num_layers - 1) {
            matrix_free(current_grad);
        }
        current_grad = grad_prev;
    }
    if (current_grad != output_grad) {
        matrix_free(current_grad);
    }
    
    return grads;
}

void gradients_free(NetworkGradients * grads){
    if(!grads){
        return;
    }
    for(int i = 0; i < grads->num_layers; i++){
        matrix_free(grads->gradients[i].dweights);
        matrix_free(grads->gradients[i].dbias);
    }
    free(grads->gradients);
    free(grads);
}
void network_update_weights(Network *net, NetworkGradients *grads, float learning_rate){
    if(!net || !grads){
        return;
    }
    for(int i = 0; i < net->num_layers; i++){
        for(int j = 0; j < net->layers[i].weights->rows * net->layers[i].weights->cols; j++){
            net->layers[i].weights->data[j] -= learning_rate * grads->gradients[i].dweights->data[j];
        }
        for(int j = 0; j < net->layers[i].bias->rows * net->layers[i].weights->cols; j++){
            
        }

    }
}