#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"


Matrix* matrix_create(int rows, int cols){
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows * cols; i++){
        m->data[i] = (float)rand() / RAND_MAX * 0.1f - 0.05f;
    }
    return m;
}

Matrix * matrix_zeros(int rows, int cols){
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(float));
    return m;  
}
void matrix_free(Matrix * m){
    free(m->data);
    free(m);
}

Matrix* matrix_multiply(Matrix *a, Matrix *b) {
    if (a->cols != b->rows) {
        printf("Error: incompatible dimensions for multiplication\n");
        return NULL;
    }
    
    Matrix *result = matrix_zeros(a->rows, b->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * b->cols + j] = sum;
        }
    }
    
    return result;
}
// Add b to a (in-place)
void matrix_add(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Error: incompatible dimensions for addition\n");
        return;
    }
    
    for (int i = 0; i < a->rows * a->cols; i++) {
        a->data[i] += b->data[i];
    }
}

Matrix* matrix_add_new(Matrix *a, Matrix *b) {
    Matrix *result = matrix_zeros(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

void matrix_relu(Matrix *m) {
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

void matrix_softmax(Matrix *m) {
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

void matrix_print(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.4f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}

