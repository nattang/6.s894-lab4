#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }
/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

namespace matmul_l1 {

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    const float *__restrict__ a, // a and b readonly
    const float *__restrict__ b,
    float *c) {

    int thread_col = threadIdx.x; // 0 to TILE_N-1
    int thread_row = threadIdx.y; // 0 to TILE_M-1
    int c_i = blockIdx.x * TILE_N + thread_col;
    int c_j = blockIdx.y * TILE_M + thread_row;

    extern __shared__ float shared[];
    float* A_shared = shared;
    float* B_shared = shared + TILE_M * TILE_K; // TODO: transpose B
 
    float sum = 0.0; // keep in register
    for (int k = 0; k < size_k; k += TILE_K) {
        // bounds checks
        if (c_j < size_j && (k + thread_col) < size_k) {
            A_shared[thread_row * TILE_K + thread_col] = a[c_j * size_k + (k + thread_col)];
        } else {
            A_shared[thread_row * TILE_K + thread_col] = 0.0f;
        }

        if ((k + thread_row) < size_k && c_i < size_i) {
            B_shared[thread_row * TILE_N + thread_col] = b[(k + thread_row) * size_i + c_i];
        } else {
            B_shared[thread_row * TILE_N + thread_col] = 0.0f;
        }
        
        __syncthreads();
        // compute
        for (int32_t tile_i = 0; tile_i < TILE_K; ++tile_i) {
            sum += A_shared[thread_row * TILE_K + tile_i] * B_shared[tile_i * TILE_N + thread_col];
        }
        __syncthreads();
    }

    if (c_i < size_i && c_j < size_j) {
        c[c_j * size_i + c_i] = sum;
    }
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 gridDim = dim3(CEIL_DIV(size_j, TILE_N), CEIL_DIV(size_i, TILE_M));
    dim3 blockDim = dim3(TILE_N, TILE_M);

    uint32_t shmem_size_bytes = (TILE_M * TILE_K * sizeof(float)) + (TILE_K * TILE_N * sizeof(float));
    // printf("shem_size in KB: %u\n", shmem_size_bytes / 1000);
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes));

    matmul_l1<<<gridDim, blockDim, shmem_size_bytes>>>(
        size_i,
        size_j,
        size_k,
        a,
        b,
        c); 
}

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

#define MICRO_TILE 4

namespace matmul_l1_reg {

__global__ void 
__launch_bounds__(TILE_M * TILE_N)
matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) 
{
    int thread_col = threadIdx.x; // 0 to TILE_N-1
    int thread_row = threadIdx.y; // 0 to TILE_M-1    

    int c_i_start  = blockIdx.x * TILE_N * MICRO_TILE + thread_col * MICRO_TILE; 
    int c_i_end = min(c_i_start + MICRO_TILE, size_j);
    int c_j_start  = blockIdx.y * TILE_M * MICRO_TILE + thread_row * MICRO_TILE;
    int c_j_end = min(c_j_start + MICRO_TILE, size_i);

    extern __shared__ float shared[];
    float* A_shared = shared; // [TILE_M * MICRO_TILE][TILE_K]
    float* B_shared = A_shared + (TILE_M * MICRO_TILE) * TILE_K; // [TILE_K][TILE_N * MICRO_TILE]

    float c_reg[MICRO_TILE][MICRO_TILE] = {0}; // store partial sum of c in registers
    float a_reg[MICRO_TILE]; 
    float b_reg[MICRO_TILE];

    for (int tile_k = 0; tile_k < size_k; tile_k += TILE_K) {
        int a_col = tile_k + thread_col;
        // load MICRO_TILE col from A
        for (int m = 0; m < MICRO_TILE; ++m) {
            int a_row = c_j_start + m; 
            // int a_col = tile_k + thread_col;
            int shmem_idx = (thread_row * MICRO_TILE + m) * TILE_K + thread_col;
            if (a_row < size_i && a_col < size_k) {
                A_shared[shmem_idx] = a[a_row * size_k + a_col];
            } else {
                A_shared[shmem_idx] = 0.0f;
            }
        }

        // load MICRO_TILE row from B
        for (int n = 0; n < MICRO_TILE; ++n) {
            int b_row = tile_k + thread_row;
            int b_col = c_i_start + n;
            int smem_idx =
                thread_row * (TILE_N * MICRO_TILE) + thread_col * MICRO_TILE + n;
            if (b_row < size_k && b_col < size_j) {
                B_shared[smem_idx] = b[b_row * size_j + b_col];
            } else {
                B_shared[smem_idx] = 0.0f;
            }
        }
        __syncthreads();

        // compute
        for (int tile_k = 0; tile_k < TILE_K; ++tile_k) {
            for (int m = 0; m < MICRO_TILE; ++m) {
                a_reg[m] = A_shared[(thread_row * MICRO_TILE + m) * TILE_K + tile_k];
            }

            for (int n = 0; n < MICRO_TILE; ++n) {
                b_reg[n] = B_shared[tile_k * (TILE_N * MICRO_TILE) + thread_col * MICRO_TILE + n];
            }

            for (int m = 0; m < MICRO_TILE; ++m) {
                for (int n = 0; n < MICRO_TILE; ++n) {
                    c_reg[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }

        __syncthreads(); 
    }

    // write back to c
    for (int m = 0; m < MICRO_TILE; ++m) {
        for (int n = 0; n < MICRO_TILE; ++n) {
            int row = c_j_start + m;
            int col = c_i_start + n;
            if (row < size_i && col < size_j) {
                c[row * size_j + col] = c_reg[m][n];
            }
        }
    }
}



void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 gridDim = dim3(CEIL_DIV(size_j, TILE_N * MICRO_TILE), CEIL_DIV(size_i, TILE_M * MICRO_TILE));
    dim3 blockDim = dim3(TILE_N, TILE_M);

    uint32_t shmem_size_bytes = 2 * TILE_M * TILE_N * MICRO_TILE * MICRO_TILE * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1_reg,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes));

    matmul_l1_reg<<<gridDim, blockDim, shmem_size_bytes>>>(
        size_i,
        size_j,
        size_k,
        a,
        b,
        c); 
}

}; // namespace matmul_l1_reg

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_k * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_k * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_i * size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_i; ++i) {
            for (int32_t j = 0; j < size_j; ++j) {
                float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
                mse += diff * diff;
                ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
            }
        }
        mse /= size_i * size_j;
        ref_mean_square /= size_i * size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
