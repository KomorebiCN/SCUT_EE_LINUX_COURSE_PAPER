/*
 * rx.c —— AI 接收端（BCH(63,36) AWGN，加权BP译码）
 * 依赖：common.h, model.h, golden_test.h
 * 用法：./rx <channel IP> <channel rx端口>
 * 编译：gcc -std=c99 -Wall -Wextra -o rx rx.c -lm
 *
 * 模型数据（Tanner图结构、权重等）由 model.h 提供
 * Golden Test 数据由 golden_test.h 提供
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <math.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "common.h"
#include "model.h"
#include "golden_test.h"

// ==================== 全局变量 ====================
static volatile sig_atomic_t keep_running = 1;

void sigint_handler(int sig) {
    (void)sig;
    keep_running = 0;
}

// ==================== 加权 BP 译码 ====================
#define CLIP_VAL     10.0f
#define ATANH_CLIP   0.9999f

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// llr: 长度为 BP_N 的输入对数似然比
// output_prob: 长度为 BP_N 的输出概率（每个比特为1的概率）
void weighted_bp_decode(const float *llr, float *output_prob) {
    // 消息数组（LLR 域）
    float msg_c2v[BP_E];  // 校验节点 -> 变量节点
    float msg_v2c[BP_E];  // 变量节点 -> 校验节点（tanh 域）

    // 初始化为 0
    memset(msg_c2v, 0, sizeof(msg_c2v));

    for (int iter = 0; iter < BP_NUM_ITERATIONS; ++iter) {
        // ----- 奇数层：变量节点 -> 校验节点 -----
        memset(msg_v2c, 0, sizeof(msg_v2c));

        for (int v = 0; v < BP_N; ++v) {
            int start = bp_var_check_offsets[v];
            int end   = bp_var_check_offsets[v + 1];
            if (start == end) continue;

            float w_v  = bp_w_input[iter][v];
            float llr_v = llr[v];

            for (int idx = start; idx < end; ++idx) {
                int c = bp_var_check_data[idx];
                int e_idx = bp_edge_idx[v][c];
                if (e_idx < 0) continue;

                float weighted_sum = 0.0f;
                int inc_start = bp_inc_offsets[e_idx];
                int inc_end   = bp_inc_offsets[e_idx + 1];
                for (int k = inc_start; k < inc_end; ++k) {
                    int e_prime_idx = bp_inc_data[2 * k];
                    int pair_idx    = bp_inc_data[2 * k + 1];
                    float w_ep = bp_edge_pair_weights[iter][pair_idx];
                    weighted_sum += w_ep * msg_c2v[e_prime_idx];
                }

                float sum = w_v * llr_v + weighted_sum;
                if (sum > CLIP_VAL)  sum = CLIP_VAL;
                if (sum < -CLIP_VAL) sum = -CLIP_VAL;
                msg_v2c[e_idx] = tanhf(0.5f * sum);
            }
        }

        // ----- 偶数层：校验节点 -> 变量节点 -----
        float new_msg_c2v[BP_E];
        memset(new_msg_c2v, 0, sizeof(new_msg_c2v));

        for (int c = 0; c < BP_M; ++c) {
            int start = bp_check_var_offsets[c];
            int end   = bp_check_var_offsets[c + 1];
            if (start == end) continue;

            int   neighbor_edges[BP_MAX_DEGREE];
            float tanh_vals[BP_MAX_DEGREE];
            int deg = 0;

            for (int idx = start; idx < end; ++idx) {
                int v = bp_check_var_data[idx];
                int e_idx = bp_edge_idx[v][c];
                if (e_idx < 0) continue;
                neighbor_edges[deg] = e_idx;
                tanh_vals[deg] = msg_v2c[e_idx];
                deg++;
            }

            for (int i = 0; i < deg; ++i) {
                int e_idx = neighbor_edges[i];
                float prod = 1.0f;
                for (int j = 0; j < deg; ++j) {
                    if (i == j) continue;
                    prod *= tanh_vals[j];
                }
                if (prod > ATANH_CLIP)  prod = ATANH_CLIP;
                if (prod < -ATANH_CLIP) prod = -ATANH_CLIP;
                new_msg_c2v[e_idx] = 2.0f * atanhf(prod);
            }
        }

        memcpy(msg_c2v, new_msg_c2v, sizeof(msg_c2v));
    }

    // ----- 最终输出层 -----
    for (int v = 0; v < BP_N; ++v) {
        float out = bp_w_out_node[v] * llr[v];
        int start = bp_var_check_offsets[v];
        int end   = bp_var_check_offsets[v + 1];
        for (int idx = start; idx < end; ++idx) {
            int c = bp_var_check_data[idx];
            int e_idx = bp_edge_idx[v][c];
            if (e_idx < 0) continue;
            out += bp_w_out_edge[v][e_idx] * msg_c2v[e_idx];
        }
        output_prob[v] = sigmoid(out);
    }
}

// ==================== Golden Test ====================
static int run_golden_test(void) {
    float output[GOLDEN_N];
    weighted_bp_decode(golden_llr, output);

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

// ==================== 主动连接 channel ====================
static int connect_to_channel(const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(fd);
        return -1;
    }

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect to channel");
        close(fd);
        return -1;
    }

    printf("已连接到 channel [%s:%d]\n", ip, port);
    return fd;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[]) {
    // Golden Test 验证模型加载正确性
    if (run_golden_test() != 0) {
        fprintf(stderr, "Golden Test 失败，请检查 model.h 数据\n");
        exit(EXIT_FAILURE);
    }

    if (argc != 3) {
        fprintf(stderr, "用法: %s <channel IP> <channel rx端口>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char *channel_ip = argv[1];
    int channel_port = atoi(argv[2]);

    if (channel_port <= 0 || channel_port > 65535) {
        fprintf(stderr, "端口号非法\n");
        exit(EXIT_FAILURE);
    }

    signal(SIGINT,  sigint_handler);
    signal(SIGTERM, sigint_handler);

    // // Golden Test 验证模型加载正确性
    // if (run_golden_test() != 0) {
    //     fprintf(stderr, "Golden Test 失败，请检查 model.h 数据\n");
    //     exit(EXIT_FAILURE);
    // }

    // 主动连接 channel
    int chan_fd = connect_to_channel(channel_ip, channel_port);
    if (chan_fd < 0) {
        exit(EXIT_FAILURE);
    }

    // 统计变量
    unsigned long total_frames  = 0;
    unsigned long total_errors  = 0;
    unsigned long total_bits    = 0;

    printf("开始接收数据...\n");

    BchFrame frame;
    while (keep_running) {
        ssize_t n = recv(chan_fd, &frame, sizeof(BchFrame), MSG_WAITALL);
        if (n <= 0) {
            if (n == 0) printf("channel 连接关闭\n");
            else perror("recv");
            break;
        }

        // 1. 将原始码字解包为 63 比特（作为标签）
        int original_bits[BCH_N];
        bytes_to_bits(frame.codeword, original_bits);

        // 2. 加权 BP 译码
        float output_prob[BCH_N];
        weighted_bp_decode(frame.llr, output_prob);

        // 3. 硬判决
        int decoded[BCH_N];
        for (int i = 0; i < BCH_N; i++) {
            decoded[i] = (output_prob[i] > 0.5f) ? 1 : 0;
        }

        // 4. 统计误码
        for (int i = 0; i < BCH_N; i++) {
            if (original_bits[i] != decoded[i]) total_errors++;
        }
        total_bits += BCH_N;
        total_frames++;

        // 定期输出
        if (total_frames % 1000 == 0) {
            double ber = (double)total_errors / total_bits;
            printf("[%lu 帧] BER: %.6e (%lu/%lu)\n",
                   total_frames, ber, total_errors, total_bits);
        }
    }

    // 最终统计
    if (total_frames > 0) {
        double ber = (double)total_errors / total_bits;
        printf("\n==================== 最终统计 ====================\n");
        printf("总帧数:        %lu\n", total_frames);
        printf("总比特数:      %lu\n", total_bits);
        printf("误比特数:      %lu\n", total_errors);
        printf("误码率 (BER):  %.6e\n", ber);
    } else {
        printf("未收到任何数据。\n");
    }

    close(chan_fd);
    return 0;
}