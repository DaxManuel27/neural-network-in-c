# Release build
release: CFLAGS += $(RELEASE_FLAGS)
release: clean $(MNIST_TEST)
	@echo "✓ Release build complete"

# Run MNIST test
run: $(MNIST_TEST)
	cd $(MNIST_DIR) && ../$(MNIST_TEST)
# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ Cleaned build artifacts"

# Clean everything including data
clean-all: clean
	cd $(MNIST_DIR) && rm -f mnist_test
	@echo "✓ Cleaned all"
#install 
# Install paths
INSTALL_PREFIX = /usr/local
INSTALL_INCLUDE = $(INSTALL_PREFIX)/include/neural_network
INSTALL_LIB = $(INSTALL_PREFIX)/lib

# Install headers and library
install: $(MNIST_TEST)
	@echo "Installing Neural Network Framework..."
	mkdir -p $(INSTALL_INCLUDE)
	mkdir -p $(INSTALL_LIB)
	cp src/*.h $(INSTALL_INCLUDE)/
	@echo "✓ Headers installed to $(INSTALL_INCLUDE)"
	@echo "✓ To use in your projects:"
	@echo "  #include <neural_network/neural_network.h>"
	@echo "  gcc yourfile.c src/*.c -o yourprogram -lm"
# Uninstall
uninstall:
	rm -rf $(INSTALL_INCLUDE)
	@echo "✓ Neural Network Framework uninstalled"

# Add to .PHONY
.PHONY: all debug release clean clean-all install uninstall help

# Help
help:
	@echo "Available targets:"
	@echo "  make              - Build release version"
	@echo "  make debug        - Build debug version with symbols"
	@echo "  make release      - Build optimized release version"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-all    - Remove all generated files"
	@echo "  make install      - Install headers to /usr/local/include"
	@echo "  make uninstall    - Uninstall headers"
	@echo "  make help         - Show this help message"

.PHONY: all debug release run clean clean-all help