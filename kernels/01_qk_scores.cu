#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef SEQ_LEN
#define SEQ_LEN 512
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

__global__ void qk_scores_kernel(const float* Q, const float* K, float* S, int seq_len, int head_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += Q[row * head_dim + d] * K[col * head_dim + d];
        }
        S[row * seq_len + col] = sum;
    }
}

int main() {
    int seq_len = SEQ_LEN;
    int head_dim = HEAD_DIM;

    size_t qk_size = seq_len * head_dim * sizeof(float);
    size_t s_size = seq_len * seq_len * sizeof(float);

    float* h_Q = (float*)malloc(qk_size);
    float* h_K = (float*)malloc(qk_size);
    float* h_S = (float*)malloc(s_size);

    for (int i = 0; i < seq_len * head_dim; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f;
        h_K[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_Q, *d_K, *d_S;
    cudaMalloc(&d_Q, qk_size);
    cudaMalloc(&d_K, qk_size);
    cudaMalloc(&d_S, s_size);

    cudaMemcpy(d_Q, h_Q, qk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qk_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    qk_scores_kernel<<<grid, block>>>(d_Q, d_K, d_S, seq_len, head_dim);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SEQ_LEN=%d Kernel execution time: %f ms\n", seq_len, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_S, d_S, s_size, cudaMemcpyDeviceToHost);

    FILE* fq = fopen("Q.bin", "wb");
    fwrite(h_Q, sizeof(float), seq_len * head_dim, fq);
    fclose(fq);

    FILE* fk = fopen("K.bin", "wb");
    fwrite(h_K, sizeof(float), seq_len * head_dim, fk);
    fclose(fk);

    FILE* fs = fopen("S.bin", "wb");
    fwrite(h_S, sizeof(float), seq_len * seq_len, fs);
    fclose(fs);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_S);
    free(h_Q); free(h_K); free(h_S);

    return 0;
}
