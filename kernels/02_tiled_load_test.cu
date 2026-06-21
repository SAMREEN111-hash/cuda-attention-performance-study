#include <cstdio>
#include <cstdlib>

#define SEQ_LEN 512
#define HEAD_DIM 64
#define TILE_SIZE 16

__global__ void tiled_load_test_kernel(const float* Q, const float* K, float* Q_out, float* K_out, int seq_len, int head_dim) {
    __shared__ float Q_tile[TILE_SIZE][HEAD_DIM];
    __shared__ float K_tile[TILE_SIZE][HEAD_DIM];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // query row this thread is "responsible" for
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // key row this thread is "responsible" for

    // Each thread loads ONE element of Q's row and ONE element of K's row into shared memory.
    // We loop over head_dim in chunks of TILE_SIZE since head_dim (64) > TILE_SIZE (16).
    for (int d = 0; d < head_dim; d += TILE_SIZE) {
        if (row < seq_len && (d + threadIdx.x) < head_dim) {
            Q_tile[threadIdx.y][d + threadIdx.x] = Q[row * head_dim + d + threadIdx.x];
        }
        if (col < seq_len && (d + threadIdx.y) < head_dim) {
            K_tile[threadIdx.x][d + threadIdx.y] = K[col * head_dim + d + threadIdx.y];
        }
    }

    __syncthreads();  // wait until ALL threads in the block finish loading before anyone reads

    // Just write shared memory straight back out, to prove the load was correct.
    // Each thread writes back its own row's data.
    if (row < seq_len && threadIdx.x < head_dim) {
        Q_out[row * head_dim + threadIdx.x] = Q_tile[threadIdx.y][threadIdx.x];
    }
    if (col < seq_len && threadIdx.y < head_dim) {
        K_out[col * head_dim + threadIdx.y] = K_tile[threadIdx.x][threadIdx.y];
    }
}
