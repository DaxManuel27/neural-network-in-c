#ifndef TRAINER_H
#define TRAINER_H

#include "framework.h"
#include "dataset.h"

typedef struct {
    int epochs;
    float learning_rate;
    int batch_size;
    int verbose;
}TrainerConfig;

typedef struct {
    float * train_loss;
    float * train_accuracy;
    float * test_loss;
    float * test_accuracy;
    int num_epochs;
}TrainingResult;

TrainingResult * train(Network * net, Dataset * train_data, Dataset * test_data, TrainerConfig * config);
 
#endif