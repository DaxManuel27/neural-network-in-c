#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

// Helper function to swap byte order (big-endian to little-endian)
uint32_t swap_endian(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8)  |
           ((val & 0x0000FF00) << 8)  |
           ((val & 0x000000FF) << 24);
}

// Load MNIST images from binary file
MNISTDataset* load_mnist_images(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Cannot open file");
        return NULL;
    }
    
    // Read magic number (should be 2051 in big-endian)
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    magic = swap_endian(magic);
    
    if (magic != 2051) {
        printf("Error: Invalid magic number %d (expected 2051)\n", magic);
        fclose(file);
        return NULL;
    }
    
    // Read number of images
    uint32_t num_images;
    fread(&num_images, sizeof(uint32_t), 1, file);
    num_images = swap_endian(num_images);
    
    // Read rows and cols (both should be 28)
    uint32_t rows, cols;
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);
    rows = swap_endian(rows);
    cols = swap_endian(cols);
    
    printf("Loading %d images (%d x %d)\n", num_images, rows, cols);
    
    // Allocate dataset
    MNISTDataset *dataset = malloc(sizeof(MNISTDataset));
    dataset->count = num_images;
    dataset->images = malloc(num_images * sizeof(MNISTImage));
    
    if (!dataset->images) {
        printf("Error: Could not allocate memory for images\n");
        fclose(file);
        free(dataset);
        return NULL;
    }
    
    // Read all images
    for (int i = 0; i < num_images; i++) {
        uint8_t pixel_bytes[784];
        size_t read = fread(pixel_bytes, sizeof(uint8_t), 784, file);
        
        if (read != 784) {
            printf("Error: Could not read image %d\n", i);
            break;
        }
        
        // Convert bytes to floats (normalize to 0-1 range)
        for (int j = 0; j < 784; j++) {
            dataset->images[i].pixels[j] = pixel_bytes[j] / 255.0f;
        }
        
        // Print progress every 10000 images
        if ((i + 1) % 10000 == 0) {
            printf("  Loaded %d images\n", i + 1);
        }
    }
    
    fclose(file);
    printf("Successfully loaded %d images\n", num_images);
    return dataset;
}

// Load MNIST labels from binary file
void load_mnist_labels(const char *filename, MNISTDataset *dataset) {
    if (!dataset) {
        printf("Error: dataset is NULL\n");
        return;
    }
    
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Cannot open file");
        return;
    }
    
    // Read magic number (should be 2049 for labels)
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    magic = swap_endian(magic);
    
    if (magic != 2049) {
        printf("Error: Invalid magic number %d (expected 2049)\n", magic);
        fclose(file);
        return;
    }
    
    // Read number of labels
    uint32_t num_labels;
    fread(&num_labels, sizeof(uint32_t), 1, file);
    num_labels = swap_endian(num_labels);
    
    if (num_labels != dataset->count) {
        printf("Warning: Number of labels (%d) doesn't match images (%d)\n", 
               num_labels, dataset->count);
    }
    
    // Read labels
    for (int i = 0; i < dataset->count; i++) {
        size_t read = fread(&dataset->images[i].label, sizeof(uint8_t), 1, file);
        if (read != 1) {
            printf("Error: Could not read label %d\n", i);
            break;
        }
    }
    
    fclose(file);
    printf("Successfully loaded %d labels\n", num_labels);
}

// Free MNIST dataset memory
void mnist_dataset_free(MNISTDataset *dataset) {
    if (dataset) {
        free(dataset->images);
        free(dataset);
    }
}