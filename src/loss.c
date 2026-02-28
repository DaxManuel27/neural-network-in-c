#include "loss.h"
#include <math.h>
#include <stdio.h>


static float cross_entropy_loss(Matrix * predictions, int true_label){
    if(!predictions || true_label < 0 || true_label >=10){
        return -0.1f;
    }
    float pred = predictions->data[true_label];
    pred = fmax(pred, 1e-7f);
    float loss = -log(pred);
    return loss;
}

static Matrix * cross_entropy_gradient(Matrix * predictions, int true_label){
    if(!predictions || true_label < 0 || true_label >= 10){
        return NULL;
    }
    Matrix * grad = matrix_zeros(1, 10);
    for (int i = 0; i < 10; i++) {
        grad->data[i] = predictions->data[i];
    }
    grad->data[true_label] -= 1.0f;
    
    return grad;
}
static float mse_loss(Matrix *predictions, int true_label) {
    if (!predictions || true_label < 0 || true_label >= 10) {
        printf("Error: Invalid predictions or label\n");
        return -1.0f;
    }
    
    float loss = 0.0f;
    
    for (int i = 0; i < 10; i++) {
        float target = (i == true_label) ? 1.0f : 0.0f;
        float diff = predictions->data[i] - target;
        loss += diff * diff;
    }
    
    return loss / 2.0f;  
}

static Matrix* mse_gradient(Matrix *predictions, int true_label) {
    if (!predictions || true_label < 0 || true_label >= 10) {
        printf("Error: Invalid predictions or label\n");
        return NULL;
    }
    
    Matrix *grad = matrix_zeros(1, 10);
    

    for (int i = 0; i < 10; i++) {
        float target = (i == true_label) ? 1.0f : 0.0f;
        grad->data[i] = predictions->data[i] - target;
    }
    
    return grad;
}

float calculate_loss_with_type(Matrix *predictions, int true_label, LossType type) {
    if (!predictions) {
        printf("Error: NULL predictions in calculate_loss_with_type\n");
        return -1.0f;
    }
    
    switch (type) {
        case LOSS_CROSS_ENTROPY:
            return cross_entropy_loss(predictions, true_label);
        case LOSS_MSE:
            return mse_loss(predictions, true_label);
        default:
            printf("Error: Unknown loss type\n");
            return -1.0f;
    }
}
Matrix* loss_gradient_with_type(Matrix *predictions, int true_label, LossType type) {
    if (!predictions) {
        printf("Error: NULL predictions in loss_gradient_with_type\n");
        return NULL;
    }
    
    switch (type) {
        case LOSS_CROSS_ENTROPY:
            return cross_entropy_gradient(predictions, true_label);
        case LOSS_MSE:
            return mse_gradient(predictions, true_label);
        default:
            printf("Error: Unknown loss type\n");
            return matrix_zeros(1, 10);
    }
}
const char* loss_name(LossType type) {
    switch (type) {
        case LOSS_CROSS_ENTROPY:
            return "Cross-Entropy";
        case LOSS_MSE:
            return "Mean Squared Error";
        default:
            return "Unknown";
    }
}

float calculate_loss(Matrix *predictions, int true_label) {
    return calculate_loss_with_type(predictions, true_label, LOSS_CROSS_ENTROPY);
}