#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

typedef enum {
    LOSS_CROSS_ENTROPY,
    LOSS_MSE
} LossType;

float calculate_loss_with_type(Matrix *predictions, int true_label, LossType type);
float calculate_loss(Matrix *predictions, int true_label);
Matrix* loss_gradient_with_type(Matrix *predictions, int true_label, LossType type);
const char* loss_name(LossType type);

#endif