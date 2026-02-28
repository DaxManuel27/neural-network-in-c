#include "train.h"
#include "network.h"

TrainingResult * train(Network * net, Dataset * train_data, Dataset * test_data, TrainerConfig * config) {
    if(!net || !train_data || !config){
        return NULL;
    }
    TrainingResult * result = malloc(sizeof(TrainingResult));
    if(!result){
        return NULL;
    }
    result->train_loss = malloc(config->epochs * sizeof(float));
    result->train_accuracy = malloc(config->epochs * sizeof(float));
    result->test_loss = malloc(config->epochs * sizeof(float));
    result->test_accuracy = malloc(config->epochs * sizeof(float));

    if (!result->train_loss || !result->train_accuracy || 
        !result->test_loss || !result->test_accuracy) {
        printf("Error: Failed to allocate metric arrays\n");
        free(result->train_loss);
        free(result->train_accuracy);
        free(result->test_loss);
        free(result->test_accuracy);
        free(result);
        return NULL;
    }
    result->num_epochs = config->epochs;
    
    //epoch loop
    for(int epoch = 0; epoch < config->epochs; epoch++){
        if(config->verbose){
            printf("Epoch %d/%d\n", epoch + 1, config->epochs);
        }
        float total_loss = 0.0f;
        int correct = 0;
        
        
        //train loop
        for(int img = 0; img < train_data->count; img++){
            Matrix * image = dataset_get_example(train_data, img);
            int label = dataset_get_label(train_data, img);

            network_forward(net, image);
            Matrix * output = net->activations[net->num_layers - 1];
            float loss = calculate_loss(output, label);
            total_loss += loss;
            Matrix * output_grad = output_gradient(output, label);
            NetworkGradients * grads = network_backward(net, image, output_grad);
            network_update_weights(net, grads, config->learning_rate);
            int predicted_digit = 0;
            float max_prob = output->data[0];
            for (int d = 1; d < net->num_classes; d++) {
                if (output->data[d] > max_prob) {
                    max_prob = output->data[d];
                    predicted_digit = d;
                }
            }

            if (predicted_digit == label) {
                correct++;
            }
            matrix_free(output_grad);
            gradients_free(grads);
            if (config->verbose && (img + 1) % 10000 == 0) {
                printf("  Processed %d/%d images\n", img + 1, train_data->count);
            }
        }
        result->train_loss[epoch] = total_loss / train_data->count;
        result->train_accuracy[epoch] = (100.0f * correct) / train_data->count;

        //test loop
        if(test_data){
            float test_total_loss = 0.0f;
            int test_correct = 0;

            for(int img = 0; img < test_data->count; img++){
                // Get image and label
                Matrix *image = dataset_get_example(test_data, img);
                int label = dataset_get_label(test_data, img);
                
                // Forward pass only (no backprop)
                network_forward(net, image);
                Matrix *output = net->activations[net->num_layers - 1];
                
                // Calculate loss
                float loss = calculate_loss(output, label);
                test_total_loss += loss;
                
                // Check accuracy
                int predicted_digit = 0;
                float max_prob = output->data[0];
                for (int d = 1; d < net->num_classes; d++) {
                    if (output->data[d] > max_prob) {
                        max_prob = output->data[d];
                        predicted_digit = d;
                    }
                }
                
                if (predicted_digit == label) {
                    test_correct++;
                }
                
                if (config->verbose && (img + 1) % 2500 == 0) {
                    printf("  Tested %d/%d images\n", img + 1, test_data->count);
                }
            }
             result->test_loss[epoch] = test_total_loss / test_data->count;
            result->test_accuracy[epoch] = (100.0f * test_correct) / test_data->count;
        }
        
    }

    


    return result;
}

void training_result_free(TrainingResult * result){
    if(!result){
        return;
    }
    free(result->train_loss);
    free(result->train_accuracy);
    free(result->test_loss);
    free(result->test_accuracy);
    free(result);
}