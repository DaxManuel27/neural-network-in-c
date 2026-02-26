#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"

typedef struct {
    Matrix **examples;
    int *labels;
    int count;
    int input_size;
} Dataset;

// Create empty dataset
Dataset* dataset_create(int count, int input_size);

// Get single example
Matrix* dataset_get_example(Dataset *dataset, int index);

// Get label
int dataset_get_label(Dataset *dataset, int index);

// Shuffle dataset
void dataset_shuffle(Dataset *dataset);

// Split dataset (for train/test split)
void dataset_split(Dataset *dataset, float split_ratio, 
                   Dataset **train, Dataset **test);

// Cleanup
void dataset_free(Dataset *dataset);

#endif