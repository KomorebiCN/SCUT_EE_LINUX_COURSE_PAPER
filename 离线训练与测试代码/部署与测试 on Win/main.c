#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "model.h"
#include "golden_test.h"

#define CLIP_VAL 10.0f
#define ATANH_CLIP 0.9999f

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// 生成标准正态分布随机数 (Box-Muller 方法)
static double gaussian_rand() {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    double u, v, s;
    do {
        u = (rand() / (RAND_MAX + 1.0)) * 2.0 - 1.0;
        v = (rand() / (RAND_MAX + 1.0)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    has_spare = 1;
    return u * s;
}

// 将 LLR 数组解码为概率输出
// llr: 长度为 BP_N 的输入对数似然比
// output_prob: 长度为 BP_N 的输出概率（每个比特为1的概率）
void weighted_bp_decode(const float *llr, float *output_prob) {
    // 消息数组（LLR 域）
    float msg_c2v[BP_E];  // 校验节点 -> 变量节点 (偶数层输出)
    float msg_v2c[BP_E];  // 变量节点 -> 校验节点 (奇数层输出，tanh 域)

    // 初始化为 0 (对应公式中 x_{0,e'}=0)
    memset(msg_c2v, 0, sizeof(msg_c2v));

    for (int iter = 0; iter < BP_NUM_ITERATIONS; ++iter) {
        // ----- 奇数层：变量节点 -> 校验节点 (公式4) -----
        // 清零当前奇数层消息
        memset(msg_v2c, 0, sizeof(msg_v2c));

        for (int v = 0; v < BP_N; ++v) {
            // 获取变量节点 v 的所有邻居校验节点
            int start = bp_var_check_offsets[v];
            int end = bp_var_check_offsets[v+1];
            if (start == end) continue;

            float w_v = bp_w_input[iter][v];
            float llr_v = llr[v];

            // 对每个相邻校验节点 c
            for (int idx = start; idx < end; ++idx) {
                int c = bp_var_check_data[idx];
                int e_idx = bp_edge_idx[v][c];
                if (e_idx < 0) continue;

                // 计算加权和：∑_{c' ≠ c} w_{i,e,e'} * msg_c2v[e'] (msg_c2v 是 LLR 域)
                float weighted_sum = 0.0f;
                int inc_start = bp_inc_offsets[e_idx];
                int inc_end = bp_inc_offsets[e_idx+1];
                for (int k = inc_start; k < inc_end; ++k) {
                    int e_prime_idx = bp_inc_data[2*k];
                    int pair_idx = bp_inc_data[2*k+1];
                    float w_ep = bp_edge_pair_weights[iter][pair_idx];
                    weighted_sum += w_ep * msg_c2v[e_prime_idx];
                }

                float sum = w_v * llr_v + weighted_sum;
                // 截断防止 tanh 饱和
                if (sum > CLIP_VAL) sum = CLIP_VAL;
                if (sum < -CLIP_VAL) sum = -CLIP_VAL;
                // 公式 (4): 输出 tanh(0.5 * sum)
                msg_v2c[e_idx] = tanhf(0.5f * sum);
            }
        }

        // ----- 偶数层：校验节点 -> 变量节点 (公式5) -----
        // 清零当前偶数层消息
        float new_msg_c2v[BP_E];
        memset(new_msg_c2v, 0, sizeof(new_msg_c2v));

        for (int c = 0; c < BP_M; ++c) {
            int start = bp_check_var_offsets[c];
            int end = bp_check_var_offsets[c+1];
            if (start == end) continue;

            // 收集所有邻居变量节点的边索引和 tanh 值
            int neighbor_edges[BP_MAX_DEGREE];
            float tanh_vals[BP_MAX_DEGREE];
            int deg = 0;
            for (int idx = start; idx < end; ++idx) {
                int v = bp_check_var_data[idx];
                int e_idx = bp_edge_idx[v][c];
                if (e_idx < 0) continue;
                neighbor_edges[deg] = e_idx;
                // msg_v2c 已经是 tanh(0.5 * ...)，直接使用
                tanh_vals[deg] = msg_v2c[e_idx];
                deg++;
            }

            // 对每个邻居变量节点计算乘积（排除自身）
            for (int i = 0; i < deg; ++i) {
                int e_idx = neighbor_edges[i];
                float prod = 1.0f;
                for (int j = 0; j < deg; ++j) {
                    if (i == j) continue;
                    prod *= tanh_vals[j];
                }
                // 避免 atanh 溢出
                if (prod > ATANH_CLIP) prod = ATANH_CLIP;
                if (prod < -ATANH_CLIP) prod = -ATANH_CLIP;
                // 公式 (5)
                new_msg_c2v[e_idx] = 2.0f * atanhf(prod);
            }
        }

        // 更新 msg_c2v 为下一轮迭代的输入
        memcpy(msg_c2v, new_msg_c2v, sizeof(msg_c2v));
    }

    // ----- 最终输出层 (公式6) -----
    for (int v = 0; v < BP_N; ++v) {
        float out = bp_w_out_node[v] * llr[v];
        int start = bp_var_check_offsets[v];
        int end = bp_var_check_offsets[v+1];
        for (int idx = start; idx < end; ++idx) {
            int c = bp_var_check_data[idx];
            int e_idx = bp_edge_idx[v][c];
            if (e_idx < 0) continue;
            out += bp_w_out_edge[v][e_idx] * msg_c2v[e_idx];
        }
        output_prob[v] = sigmoid(out);
    }
}

void golden_test() {
    float output[GOLDEN_N];
    weighted_bp_decode(golden_llr, output);  // 调用正式译码函数

    float max_err = 0.0f;
    for (int i = 0; i < GOLDEN_N; i++) {
        float err = fabsf(output[i] - golden_output[i]);
        if (err > max_err) max_err = err;
    }

    printf("===== Golden Test =====\n");
    printf("Max error: %e\n", max_err);
    if (max_err < 1e-5f) {
        printf("Golden test passed!\n");
        printf("=======================\n");
        return 0;
    } else {
        printf("Golden test failed!\n");
        printf("=======================\n");
        return -1;
    }
}

void generate_random_codeword(int *codeword) {
    int msg[BP_K];

    // 1. 生成随机的信息比特 (0 或 1)
    for (int i = 0; i < BP_K; ++i) {
        msg[i] = rand() % 2;
    }

    // 2. 编码: codeword = msg * G (mod 2)
    // 初始化码字全 0
    memset(codeword, 0, sizeof(int) * BP_N);

    for (int i = 0; i < BP_K; ++i) {
        if (msg[i] == 0) continue;   // 跳过 0，减少不必要的循环
        for (int j = 0; j < BP_N; ++j) {
            // 模2加法 (异或)
            codeword[j] ^= bp_G[i][j];
        }
    }
}

// 将二进制码字通过 AWGN 信道，输出 LLR
// codeword: 长度为 n 的 0/1 数组
// llr: 输出数组，长度 n，存储 LLR
// snr_db: 信噪比 (dB)
// n: 码字长度
void awgn_channel_llr(const int *codeword, float *llr, double snr_db, int n) {
    // BPSK 映射: 0 -> +1, 1 -> -1
    double snr_linear = pow(10.0, snr_db / 10.0);
    double sigma2 = 1.0 / (2.0 * snr_linear);
    double sigma = sqrt(sigma2);

    for (int i = 0; i < n; ++i) {
        double symbol = (codeword[i] == 0) ? 1.0 : -1.0;
        double noise = gaussian_rand() * sigma;
        double received = symbol + noise;
        double llr_val = 2.0 * received / sigma2;
        llr[i] = (float)llr_val;
    }
}

// ---------- 统计误码率 ----------
double compute_ber(const int *original, const int *decoded, int n) {
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (original[i] != decoded[i]) errors++;
    }
    return (double)errors / n;
}

int main(void) {
    golden_test();

    // 测试的 SNR 列表 (dB)
    double snr_list[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10};
    int num_snr = sizeof(snr_list) / sizeof(snr_list[0]);

    // 每个 SNR 下测试的码字数
    int num_trials = 10000;   // 可根据需要调整，越大 BER 越精确

    // 临时数组
    int original[BP_N];
    float llr[BP_N];
    float output_prob[BP_N];
    int decoded[BP_N];

    printf("SNR (dB)\tBER\n");
    printf("-----------------\n");
    srand(time(NULL));
    for (int s = 0; s < num_snr; ++s) {
        double snr_db = snr_list[s];
        int total_bits = 0;
        int total_errors = 0;

        for (int trial = 0; trial < num_trials; ++trial) {
            // 1. 生成随机码字
            generate_random_codeword(original);

            // 2. 通过 AWGN 信道得到 LLR
            awgn_channel_llr(original, llr, snr_db, BP_N);

            // 3. BP 解码，得到输出概率
            weighted_bp_decode(llr, output_prob);

            // 4. 硬判决 (概率 > 0.5 判为 1)
            for (int i = 0; i < BP_N; ++i) {
                decoded[i] = (output_prob[i] > 0.5f) ? 1 : 0;
            }

            // 5. 统计错误
            for (int i = 0; i < BP_N; ++i) {
                if (original[i] != decoded[i]) total_errors++;
            }
            total_bits += BP_N;
        }

        double ber = (double)total_errors / total_bits;
        printf("%.1f\t\t%e\t\t%d/%d\n", snr_db, ber,total_errors,total_bits);
    }


    return 0;
}