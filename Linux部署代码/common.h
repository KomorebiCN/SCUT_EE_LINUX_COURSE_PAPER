/*
 * common.h —— BCH(63,36) + AWGN 仿真公共头文件
 * 被 tx.c / channel.c / rx.c 共享
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <math.h>

// ==================== BCH(63, 36) 参数 ====================
#define BCH_N   63    // 码字长度（比特）
#define BCH_K   36    // 信息位长度（比特）
#define BCH_M   27    // 校验位长度 = N - K

// 码字按字节打包所需字节数：63 位 = 7 完整字节 + 1 字节只用低 7 位
#define BCH_BYTES ((BCH_N + 7) / 8)   // = 8

// ==================== 通信帧结构 ====================
// 发送端 → 信道模拟器：原始码字字节流
// 信道模拟器 → 接收端：原始码字字节流 + 浮点 LLR 数组
typedef struct {
    uint8_t codeword[BCH_BYTES];     // 原始码字（打包比特，接收端用于标签）
    float   llr[BCH_N];              // AWGN 信道输出的对数似然比
} BchFrame;

// ==================== 比特 ⇄ 字节 转换工具 ====================

// 将 63 位比特数组打包成 8 字节（低位在前）
static inline void bits_to_bytes(const int bits[BCH_N], uint8_t bytes[BCH_BYTES]) {
    for (int i = 0; i < BCH_BYTES; i++) bytes[i] = 0;
    for (int i = 0; i < BCH_N; i++) {
        if (bits[i])
            bytes[i / 8] |= (1 << (i % 8));
    }
}

// 将 8 字节解包为 63 位比特数组（低位在前）
static inline void bytes_to_bits(const uint8_t bytes[BCH_BYTES], int bits[BCH_N]) {
    for (int i = 0; i < BCH_N; i++) {
        bits[i] = (bytes[i / 8] >> (i % 8)) & 1;
    }
}

// ==================== 误码率统计 ====================

// 比较原始码字与译码结果，返回误码率
static inline double compute_ber(const int original[BCH_N], const int decoded[BCH_N]) {
    int errors = 0;
    for (int i = 0; i < BCH_N; i++) {
        if (original[i] != decoded[i]) errors++;
    }
    return (double)errors / BCH_N;
}

#endif /* COMMON_H */