#include "dataset.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


Dataset* dataset_create(int count, int input_size) {
    if (count <= 0 || input_size <= 0) {
        printf("Error: Invalid count or input_size\n");
        return NULL;
    }
    
    Dataset *dataset = malloc(sizeof(Dataset));
    if (!dataset) {
        printf("Error: Failed to allocate memory for dataset\n");
        return NULL;
    }
    
    dataset->examples = malloc(count * sizeof(Matrix *));
    if (!dataset->examples) {
        printf("Error: Failed to allocate memory for examples\n");
        free(dataset);
        return NULL;
    }
    
    dataset->labels = malloc(count * sizeof(int));
    if (!dataset->labels) {
        printf("Error: Failed to allocate memory for labels\n");
        free(dataset->examples);
        free(dataset);
        return NULL;
    }
    
    dataset->count = count;
    dataset->input_size = input_size;
    
 
    for (int i = 0; i < count; i++) {
        dataset->examples[i] = NULL;
        dataset->labels[i] = 0;
    }
    
    return dataset;
}


Matrix* dataset_get_example(Dataset *dataset, int index) {
    if (!dataset || index < 0 || index >= dataset->count) {
        printf("Error: Invalid dataset or index\n");
        return NULL;
    }
    
    return dataset->examples[index];
}


int dataset_get_label(Dataset *dataset, int index) {
    if (!dataset || index < 0 || index >= dataset->count) {
        printf("Error: Invalid dataset or index\n");
        return -1;
    }
    
    return dataset->labels[index];
}


void dataset_shuffle(Dataset *dataset) {
    if (!dataset) {
        printf("Error: NULL dataset in dataset_shuffle\n");
        return;
    }
    

    srand(time(NULL));
    
    for (int i = dataset->count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        Matrix *temp_example = dataset->examples[i];
        dataset->examples[i] = dataset->examples[j];
        dataset->examples[j] = temp_example;
        

        int temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }
}


void dataset_split(Dataset *dataset, float split_ratio, 
                   Dataset **train, Dataset **test) {
    if (!dataset || !train || !test || split_ratio <= 0.0f || split_ratio >= 1.0f) {
        printf("Error: Invalid arguments in dataset_split\n");
        return;
    }
    
    int train_count = (int)(dataset->count * split_ratio);
    int test_count = dataset->count - train_count;
    

    *train = dataset_create(train_count, dataset->input_size);
    *test = dataset_create(test_count, dataset->input_size);
    
    if (!*train || !*test) {
        printf("Error: Failed to create train/test datasets\n");
        return;
    }
    

    for (int i = 0; i < train_count; i++) {
        (*train)->examples[i] = dataset->examples[i];
        (*train)->labels[i] = dataset->labels[i];
    }
    
  
    for (int i = 0; i < test_count; i++) {
        (*test)->examples[i] = dataset->examples[train_count + i];
        (*test)->labels[i] = dataset->labels[train_count + i];
    }
}


void dataset_free(Dataset *dataset) {
    if (!dataset) return;
    

    for (int i = 0; i < dataset->count; i++) {
        if (dataset->examples[i]) {
            matrix_free(dataset->examples[i]);
        }
    }
    
    free(dataset->examples);
    free(dataset->labels);
    free(dataset);
}