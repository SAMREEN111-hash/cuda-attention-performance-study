#include <cstdio>
#include <cstdlib>

#define SEQ_LEN 512
#define HEAD_DIM 64
#define TILE_SIZE 16

__global__ void tiled_load_test_kernel(const float* Q, const float* K, float* Q_out, float* K_out, int seq_len, int head_dim) {
    __shared__ float Q_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float K_tile[TILE_SIZE][HEAD_DIM];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    for (int d = 0; d < head_dim; d += TILE_SIZE) {
        if (row < seq_len && (d + threadIdx.x) < head_dim) {
            Q_tile[threadIdx.y][d + threadIdx.x] = Q[row * head_dim + d + threadIdx.x];
        }
        if (col < seq_len && (d + threadIdx.y) < head_dim) {
            K_tile[threadIdx.x][d + threadIdx.y] = K[col * head_dim + d + threadIdx.y];
        }
    }

    __syncthreads();

    for (int d = 0; d < head_dim; d += TILE_SIZE) {
        if (row < seq_len && (d + threadIdx.x) < head_dim) {
            Q_out[row * head_dim + d + threadIdx.x] = Q_tile[threadIdx.y][d + threadIdx.x];
        }
        if (col < seq_len && (d + threadIdx.y) < head_dim) {
            K_out[col * head_dim + d + threadIdx.y] = K_tile[threadIdx.x][d + threadIdx.y];
        }
    }
}

int main() {
    int seq_len = SEQ_LEN;
    int head_dim = HEAD_DIM;
    size_t qk_size = seq_len * head_dim * sizeof(float);

    float* h_Q = (float*)malloc(qk_size);
    float* h_K = (float*)malloc(qk_size);
    float* h_Q_out = (float*)malloc(qk_size);
    float* h_K_out = (float*)malloc(qk_size);

    for (int i = 0; i < seq_len * head_dim; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f;
        h_K[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_Q, *d_K, *d_Q_out, *d_K_out;
    cudaMalloc(&d_Q, qk_size);
    cudaMalloc(&d_K, qk_size);
    cudaMalloc(&d_Q_out, qk_size);
    cudaMalloc(&d_K_out, qk_size);

    cudaMemcpy(d_Q, h_Q, qk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qk_size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((seq_len + TILE_SIZE - 1) / TILE_SIZE, (seq_len + TILE_SIZE - 1) / TILE_SIZE);

    tiled_load_test_kernel<<<grid, block>>>(d_Q, d_K, d_Q_out, d_K_out, seq_len, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Q_out, d_Q_out, qk_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K_out, d_K_out, qk_size, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int i = 0; i < seq_len * head_dim; i++) {
        if (h_Q[i] != h_Q_out[i]) mismatches++;
        if (h_K[i] != h_K_out[i]) mismatches++;
    }

    printf("Total mismatches: %d (out of %d values checked)\n", mismatches, seq_len * head_dim * 2);
    if (mismatches == 0) {
        printf("PASS: shared memory load/store is correct\n");
    } else {
        printf("FAIL: shared memory load/store has bugs\n");
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_Q_out); cudaFree(d_K_out);
    free(h_Q); free(h_K); free(h_Q_out); free(h_K_out);

    return 0;
}
