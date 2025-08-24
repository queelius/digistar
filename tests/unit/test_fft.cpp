#include <iostream>
#include <vector>
#include <complex>
#include "src/algorithms/FastFFT2D.h"

int main() {
    std::cout << "Testing custom FFT implementation\n";
    
    // Test small size first
    size_t size = 64;
    
    try {
        FastFFT2D fft(size);
        std::cout << "FFT object created for size " << size << "\n";
        
        // Create test data
        std::vector<float> real_data(size * size, 1.0f);
        std::vector<std::complex<float>> fft_data(size * size);
        
        std::cout << "Running forward FFT...\n";
        fft.forward2D_r2c(real_data.data(), fft_data.data());
        std::cout << "Forward FFT complete\n";
        
        std::cout << "Running inverse FFT...\n";
        std::vector<float> result(size * size);
        fft.inverse2D_c2r(fft_data.data(), result.data());
        std::cout << "Inverse FFT complete\n";
        
        // Check if we get back original (should be all 1s)
        float error = 0;
        for (size_t i = 0; i < size * size; i++) {
            error += std::abs(result[i] - real_data[i]);
        }
        std::cout << "Round-trip error: " << error << "\n";
        
        if (error < 0.001f) {
            std::cout << "✓ FFT working correctly!\n";
        } else {
            std::cout << "✗ FFT has errors\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
    }
    
    return 0;
}