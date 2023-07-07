#pragma GCC optimize(3)
#define INPUT_DIM 100
#define FILTER_DIM 5
#define CONV_OUT_DIM 20
#define CONV_LAYER_SIZE 10
#define OUT_NEURON_DIM 4000
#define OUT_LAYER_SIZE 10
#define OUT_LAYER_THREAD_NUM 32
#define NEG_TO_ZERO(x) (x = (x < 0 ? 0.0 : x))

extern "C" __global__ void convolution_relu_layer(
    double input[INPUT_DIM][INPUT_DIM],
    double filter[CONV_LAYER_SIZE][FILTER_DIM][FILTER_DIM],
    double output[CONV_LAYER_SIZE][CONV_OUT_DIM][CONV_OUT_DIM]
) {
    double ans = 0;

    int curr_filter = blockIdx.x; /* 0 - 9 */
    int curr_x = threadIdx.x * FILTER_DIM; /* 0 - 19 */
    int curr_y = threadIdx.y * FILTER_DIM; /* 0 - 19 */

    /* conv */
    #pragma unroll
    for (int i = 0; i < FILTER_DIM; i++) {
        #pragma unroll
        for (int j = 0; j < FILTER_DIM; j++) {
            /* dot product */
            ans += filter[curr_filter][i][j] * input[curr_x + i][curr_y + j];
        }
    }

    /* relu */
    output[curr_filter][curr_x / FILTER_DIM][curr_y / FILTER_DIM] = NEG_TO_ZERO(ans);
}

extern "C" __global__ void output_layer(
    double input[OUT_NEURON_DIM],
    double weight[OUT_LAYER_SIZE][OUT_NEURON_DIM],
    double output[OUT_LAYER_SIZE][OUT_LAYER_THREAD_NUM]
) {
    double ans = 0;

    int curr_weight = blockIdx.x; /* 0 - 9 */
    int curr_thread = threadIdx.x * 125; /* 0 - 31 */

    #pragma unroll
    for (int i = 0; i < 125; i++) {
        // maybe use 64 threads and 'unlikely()' to check for overflow would be better
        int curr_input = i + curr_thread;
        ans += input[curr_input] * weight[curr_weight][curr_input];
    }

    output[curr_weight][curr_thread / 125] = ans;
}