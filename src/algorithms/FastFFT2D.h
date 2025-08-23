#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

// Custom high-performance 2D FFT for power-of-2 grids
// Optimized specifically for PM algorithm use case
class FastFFT2D {
private:
    size_t size;  // Grid size (must be power of 2)
    size_t log2_size;
    
    // Twiddle factors (precomputed)
    std::vector<std::complex<float>> twiddle_factors;
    std::vector<size_t> bit_reverse_table;
    
    // Check if power of 2
    static bool isPowerOf2(size_t n) {
        return n && !(n & (n - 1));
    }
    
    // Compute log2
    static size_t computeLog2(size_t n) {
        size_t log = 0;
        while (n > 1) {
            n >>= 1;
            log++;
        }
        return log;
    }
    
    // Initialize bit reversal table
    void initBitReverseTable() {
        bit_reverse_table.resize(size);
        for (size_t i = 0; i < size; i++) {
            size_t rev = 0;
            size_t n = i;
            for (size_t j = 0; j < log2_size; j++) {
                rev = (rev << 1) | (n & 1);
                n >>= 1;
            }
            bit_reverse_table[i] = rev;
        }
    }
    
    // Initialize twiddle factors
    void initTwiddleFactors() {
        twiddle_factors.resize(size);
        for (size_t i = 0; i < size; i++) {
            float angle = -2.0f * M_PI * i / size;
            twiddle_factors[i] = std::complex<float>(cos(angle), sin(angle));
        }
    }
    
    // Radix-2 Cooley-Tukey FFT (in-place)
    void fft1D(std::complex<float>* data, bool inverse) {
        // Bit reversal
        for (size_t i = 0; i < size; i++) {
            size_t j = bit_reverse_table[i];
            if (i < j) {
                std::swap(data[i], data[j]);
            }
        }
        
        // Cooley-Tukey FFT
        for (size_t stage = 1; stage <= log2_size; stage++) {
            size_t m = 1 << stage;  // 2^stage
            size_t m2 = m >> 1;      // m/2
            
            for (size_t k = 0; k < size; k += m) {
                for (size_t j = 0; j < m2; j++) {
                    size_t idx1 = k + j;
                    size_t idx2 = idx1 + m2;
                    
                    // Twiddle factor
                    size_t twiddle_idx = (j * size) / m;
                    std::complex<float> w = twiddle_factors[twiddle_idx];
                    if (inverse) w = std::conj(w);
                    
                    // Butterfly operation
                    std::complex<float> t = w * data[idx2];
                    data[idx2] = data[idx1] - t;
                    data[idx1] = data[idx1] + t;
                }
            }
        }
        
        // Normalize for inverse transform
        if (inverse) {
            float norm = 1.0f / size;
            for (size_t i = 0; i < size; i++) {
                data[i] *= norm;
            }
        }
    }
    
    // AVX2 optimized version for real-to-complex transform
    void fft1D_AVX2_r2c(float* real_in, std::complex<float>* complex_out) {
        // Pack real data into complex (imaginary = 0)
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            complex_out[i] = std::complex<float>(real_in[i], 0);
        }
        
        // Perform complex FFT
        fft1D(complex_out, false);
    }
    
public:
    FastFFT2D(size_t grid_size) : size(grid_size) {
        if (!isPowerOf2(size)) {
            throw std::runtime_error("FFT size must be power of 2");
        }
        
        log2_size = computeLog2(size);
        initBitReverseTable();
        initTwiddleFactors();
    }
    
    // 2D FFT for real input (used in PM algorithm)
    void forward2D_r2c(float* real_data, std::complex<float>* fft_data) {
        // FFT along rows
        #pragma omp parallel for
        for (size_t y = 0; y < size; y++) {
            // Thread-local temporary buffer
            std::vector<std::complex<float>> temp(size);
            // Copy row to temp buffer
            for (size_t x = 0; x < size; x++) {
                temp[x] = std::complex<float>(real_data[y * size + x], 0);
            }
            
            // 1D FFT
            fft1D(temp.data(), false);
            
            // Copy back
            for (size_t x = 0; x < size; x++) {
                fft_data[y * size + x] = temp[x];
            }
        }
        
        // FFT along columns
        #pragma omp parallel for
        for (size_t x = 0; x < size; x++) {
            std::vector<std::complex<float>> temp(size);
            // Copy column to temp buffer
            for (size_t y = 0; y < size; y++) {
                temp[y] = fft_data[y * size + x];
            }
            
            // 1D FFT
            fft1D(temp.data(), false);
            
            // Copy back
            for (size_t y = 0; y < size; y++) {
                fft_data[y * size + x] = temp[y];
            }
        }
    }
    
    // 2D inverse FFT to real output
    void inverse2D_c2r(std::complex<float>* fft_data, float* real_data) {
        // Inverse FFT along rows
        #pragma omp parallel for
        for (size_t y = 0; y < size; y++) {
            std::vector<std::complex<float>> temp(size);
            // Copy row to temp buffer
            for (size_t x = 0; x < size; x++) {
                temp[x] = fft_data[y * size + x];
            }
            
            // 1D inverse FFT
            fft1D(temp.data(), true);
            
            // Copy back
            for (size_t x = 0; x < size; x++) {
                fft_data[y * size + x] = temp[x];
            }
        }
        
        // Inverse FFT along columns and extract real part
        #pragma omp parallel for
        for (size_t x = 0; x < size; x++) {
            std::vector<std::complex<float>> temp(size);
            // Copy column to temp buffer
            for (size_t y = 0; y < size; y++) {
                temp[y] = fft_data[y * size + x];
            }
            
            // 1D inverse FFT
            fft1D(temp.data(), true);
            
            // Extract real part
            for (size_t y = 0; y < size; y++) {
                real_data[y * size + x] = temp[y].real();
            }
        }
    }
    
    // Optimized version using AVX2 for butterfly operations
    void fft1D_AVX2(std::complex<float>* data, bool inverse) {
        // Bit reversal
        for (size_t i = 0; i < size; i++) {
            size_t j = bit_reverse_table[i];
            if (i < j) {
                std::swap(data[i], data[j]);
            }
        }
        
        // Cooley-Tukey FFT with AVX2
        for (size_t stage = 1; stage <= log2_size; stage++) {
            size_t m = 1 << stage;
            size_t m2 = m >> 1;
            
            #pragma omp parallel for
            for (size_t k = 0; k < size; k += m) {
                // Process 4 butterflies at once with AVX2
                size_t j = 0;
                for (; j + 3 < m2; j += 4) {
                    // Load 4 complex pairs
                    __m256 a_real = _mm256_setr_ps(
                        data[k+j].real(), data[k+j+1].real(), 
                        data[k+j+2].real(), data[k+j+3].real(),
                        data[k+j+m2].real(), data[k+j+m2+1].real(),
                        data[k+j+m2+2].real(), data[k+j+m2+3].real()
                    );
                    __m256 a_imag = _mm256_setr_ps(
                        data[k+j].imag(), data[k+j+1].imag(),
                        data[k+j+2].imag(), data[k+j+3].imag(),
                        data[k+j+m2].imag(), data[k+j+m2+1].imag(),
                        data[k+j+m2+2].imag(), data[k+j+m2+3].imag()
                    );
                    
                    // Twiddle factors
                    __m256 w_real = _mm256_setr_ps(
                        twiddle_factors[(j*size)/m].real(),
                        twiddle_factors[((j+1)*size)/m].real(),
                        twiddle_factors[((j+2)*size)/m].real(),
                        twiddle_factors[((j+3)*size)/m].real(),
                        twiddle_factors[(j*size)/m].real(),
                        twiddle_factors[((j+1)*size)/m].real(),
                        twiddle_factors[((j+2)*size)/m].real(),
                        twiddle_factors[((j+3)*size)/m].real()
                    );
                    __m256 w_imag = _mm256_setr_ps(
                        twiddle_factors[(j*size)/m].imag(),
                        twiddle_factors[((j+1)*size)/m].imag(),
                        twiddle_factors[((j+2)*size)/m].imag(),
                        twiddle_factors[((j+3)*size)/m].imag(),
                        twiddle_factors[(j*size)/m].imag(),
                        twiddle_factors[((j+1)*size)/m].imag(),
                        twiddle_factors[((j+2)*size)/m].imag(),
                        twiddle_factors[((j+3)*size)/m].imag()
                    );
                    
                    if (inverse) {
                        w_imag = _mm256_sub_ps(_mm256_setzero_ps(), w_imag);
                    }
                    
                    // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                    __m256 t_real = _mm256_sub_ps(
                        _mm256_mul_ps(_mm256_permute2f128_ps(a_real, a_real, 0x11), w_real),
                        _mm256_mul_ps(_mm256_permute2f128_ps(a_imag, a_imag, 0x11), w_imag)
                    );
                    __m256 t_imag = _mm256_add_ps(
                        _mm256_mul_ps(_mm256_permute2f128_ps(a_real, a_real, 0x11), w_imag),
                        _mm256_mul_ps(_mm256_permute2f128_ps(a_imag, a_imag, 0x11), w_real)
                    );
                    
                    // Butterfly: top = top + t, bottom = top - t
                    __m256 top_real = _mm256_permute2f128_ps(a_real, a_real, 0x00);
                    __m256 top_imag = _mm256_permute2f128_ps(a_imag, a_imag, 0x00);
                    
                    __m256 out1_real = _mm256_add_ps(top_real, t_real);
                    __m256 out1_imag = _mm256_add_ps(top_imag, t_imag);
                    __m256 out2_real = _mm256_sub_ps(top_real, t_real);
                    __m256 out2_imag = _mm256_sub_ps(top_imag, t_imag);
                    
                    // Store results
                    float real_vals[8], imag_vals[8];
                    _mm256_storeu_ps(real_vals, _mm256_blend_ps(out1_real, out2_real, 0xF0));
                    _mm256_storeu_ps(imag_vals, _mm256_blend_ps(out1_imag, out2_imag, 0xF0));
                    
                    for (int i = 0; i < 4; i++) {
                        data[k+j+i] = std::complex<float>(real_vals[i], imag_vals[i]);
                        data[k+j+m2+i] = std::complex<float>(real_vals[i+4], imag_vals[i+4]);
                    }
                }
                
                // Handle remaining butterflies
                for (; j < m2; j++) {
                    size_t idx1 = k + j;
                    size_t idx2 = idx1 + m2;
                    
                    size_t twiddle_idx = (j * size) / m;
                    std::complex<float> w = twiddle_factors[twiddle_idx];
                    if (inverse) w = std::conj(w);
                    
                    std::complex<float> t = w * data[idx2];
                    data[idx2] = data[idx1] - t;
                    data[idx1] = data[idx1] + t;
                }
            }
        }
        
        // Normalize for inverse
        if (inverse) {
            float norm = 1.0f / size;
            __m256 norm_vec = _mm256_set1_ps(norm);
            
            size_t i = 0;
            for (; i + 3 < size; i += 4) {
                __m256 vals = _mm256_setr_ps(
                    data[i].real(), data[i].imag(),
                    data[i+1].real(), data[i+1].imag(),
                    data[i+2].real(), data[i+2].imag(),
                    data[i+3].real(), data[i+3].imag()
                );
                vals = _mm256_mul_ps(vals, norm_vec);
                float results[8];
                _mm256_storeu_ps(results, vals);
                for (int j = 0; j < 4; j++) {
                    data[i+j] = std::complex<float>(results[j*2], results[j*2+1]);
                }
            }
            for (; i < size; i++) {
                data[i] *= norm;
            }
        }
    }
};