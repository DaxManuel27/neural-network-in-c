#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    float *data;
    int rows;
    int cols;
}Matrix;

Matrix * matrix_create(int rows, int cols);
Matrix* matrix_zeros(int rows, int cols);
void matrix_free(Matrix *m);

// Operations
Matrix* matrix_multiply(Matrix *a, Matrix *b);
void matrix_add(Matrix *a, Matrix *b);
Matrix* matrix_add_new(Matrix *a, Matrix *b);

// Activations
void matrix_relu(Matrix *m);
Matrix* matrix_relu_derivative(Matrix *m);
void matrix_softmax(Matrix *m);

// Utility
void matrix_print(Matrix *m);

#endif