CC = gcc
CFLAGS = -Wall -Wextra -lm
SRC = src/matrix.c src/network.c src/activations.c src/loss.c
HEADERS = src/matrix.h src/network.h src/framework.h

mnist_example: examples/mnist.c $(SRC) $(HEADERS)
	$(CC) -o mnist_example examples/mnist.c $(SRC) $(CFLAGS)

clean:
	rm -f mnist_example

run: mnist_example
	./mnist_example