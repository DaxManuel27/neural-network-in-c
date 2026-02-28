#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
typedef struct {
    float pixels[784];
    uint8_t label;
}MNISTImage;

typedef struct {
    MNISTImage * images;
    int count;
}MNISTDataset;

MNISTDataset* load_mnist_images(const char *filename);
void load_mnist_labels(const char *filename, MNISTDataset *dataset);
void mnist_dataset_free(MNISTDataset *dataset);

#endif