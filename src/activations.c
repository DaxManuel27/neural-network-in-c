#include "activations.h"
#include <math.h>
#include <stdio.h>


static void relu_activate(Matrix *m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        if (m->data[i] < 0) m->data[i] = 0;
    }
}
Matrix* matrix_relu_derivative(Matrix *m) {
    Matrix *deriv = matrix_zeros(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        deriv->data[i] = (m->data[i] > 0) ? 1.0f : 0.0f;
    }
    return deriv;
}

static void sigmoid_activate(Matrix * m){
    for(int i = 0; i < m->rows * m->cols; i++){
        m->data[i] = 1.0f / (1.0f + exp(-m->data[i]));
    }
}
Matrix * sigmoid_derivative(Matrix * m){
    Matrix *deriv = matrix_zeros(m->rows, m->cols);
    for(int i = 0; i < m->rows * m->cols; i++){
        float sig = 1.0f / (1.0f + exp(-m->data[i]));
        deriv->data[i] = sig * (1.0f - sig);
    }
    return deriv;
}
static void softmax_activate(Matrix *m) {
    int n = m->rows * m->cols;

    float max_val = m->data[0];
    for (int i = 1; i < n; i++) {
        if (m->data[i] > max_val) max_val = m->data[i];
    }
    

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        m->data[i] = exp(m->data[i] - max_val);
        sum += m->data[i];
    }
    

    for (int i = 0; i < n; i++) {
        m->data[i] /= sum;
    }
}
static Matrix * softmax_derivative(Matrix * m){
    return matrix_zeros(m->rows, m->cols);
}
static void tanh_activate(Matrix * m){
    for(int i = 0; i < m-> rows * m->cols; i++){
        m->data[i] = tanh(m->data[i]);
    }
}
static Matrix * tanh_derivative(Matrix * m){
    Matrix * deriv = matrix_zeros(m->rows, m->cols);
    for(int i = 0; i < m->rows * m->cols; i++){
        float t = tanh(m->data[i]);
        deriv->data[i] = 1.0f - (t * t);
    }
    return deriv;
}


void apply_activation(Matrix * m, ActivationType type){
    if(!m){
        return;
    }
    switch(type){
        case ACTIVATION_RELU:
            relu_activate(m);
            break;
        case ACTIVATION_SIGMOID:
            sigmoid_activate(m);
            break;
        case ACTIVATION_SOFTMAX:
            softmax_activate(m);
            break;
        case ACTIVATION_TANH:
            tanh_activate(m);
            break;
    }
}

Matrix* activation_derivative(Matrix *m, ActivationType type) {
    if (!m) {
        return NULL;
    }
    
    switch (type) {
        case ACTIVATION_SIGMOID:
            return sigmoid_derivative(m);
        case ACTIVATION_TANH:
            return tanh_derivative(m);
        case ACTIVATION_SOFTMAX:
            return softmax_derivative(m);
        default:
            return matrix_zeros(m->rows, m->cols);
    }
}