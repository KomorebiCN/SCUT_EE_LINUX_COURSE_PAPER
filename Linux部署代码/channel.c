/*
 * channel.c —— 信道模拟器（AWGN，多线程，动态重连）
 * 依赖：common.h
 * 用法：./channel <tx监听端口> <rx监听端口> [SNR(dB)]
 * 编译：gcc -std=c99 -Wall -Wextra -pthread -o channel channel.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>
#include <math.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "common.h"

// ==================== 全局状态 ====================
static volatile sig_atomic_t keep_running = 1;
static int tx_listen_fd = -1;
static int rx_listen_fd = -1;

static int current_tx_fd = -1;
static int current_rx_fd = -1;

static pthread_mutex_t conn_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  tx_connected = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  session_end  = PTHREAD_COND_INITIALIZER;

// 全局参数
static struct {
    int    tx_port;
    int    rx_port;
    double snr_db;
} args;

// ==================== 信号处理 ====================
void sigint_handler(int sig) {
    (void)sig;
    keep_running = 0;
    if (tx_listen_fd >= 0) shutdown(tx_listen_fd, SHUT_RDWR);
    if (rx_listen_fd >= 0) shutdown(rx_listen_fd, SHUT_RDWR);
    pthread_cond_broadcast(&tx_connected);
}

// ==================== 参数解析 ====================
static void parse_arguments(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "用法: %s <tx监听端口> <rx监听端口> [SNR(dB)]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    args.tx_port = atoi(argv[1]);
    args.rx_port = atoi(argv[2]);
    args.snr_db  = (argc == 4) ? atof(argv[3]) : 3.0;

    if (args.tx_port <= 0 || args.tx_port > 65535 ||
        args.rx_port <= 0 || args.rx_port > 65535) {
        fprintf(stderr, "参数非法\n");
        exit(EXIT_FAILURE);
    }
}

// ==================== 监听套接字 ====================
static int create_listen_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(EXIT_FAILURE); }

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(fd);
        exit(EXIT_FAILURE);
    }
    if (listen(fd, 1) < 0) {
        perror("listen");
        close(fd);
        exit(EXIT_FAILURE);
    }
    printf("监听端口 %d 已就绪\n", port);
    return fd;
}

// ==================== 高斯随机数 ====================
static double gaussian_rand(unsigned int *seed) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    double u, v, s;
    do {
        u = (rand_r(seed) / (double)RAND_MAX) * 2.0 - 1.0;
        v = (rand_r(seed) / (double)RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    has_spare = 1;
    return u * s;
}

// ==================== AWGN 信道仿真 ====================
// 输入：63位码字比特数组，输出：63个LLR浮点数
static void awgn_channel(const int cw_bits[BCH_N], float llr[BCH_N],
                         double snr_db, unsigned int *seed) {
    double snr_linear = pow(10.0, snr_db / 10.0);
    double sigma2 = 1.0 / (2.0 * snr_linear);
    double sigma  = sqrt(sigma2);

    for (int i = 0; i < BCH_N; i++) {
        double symbol = (cw_bits[i] == 0) ? 1.0 : -1.0;
        double noise  = gaussian_rand(seed) * sigma;
        double received = symbol + noise;
        llr[i] = (float)(2.0 * received / sigma2);
    }
}

// ==================== 连接接收线程 ====================
static void *tx_acceptor(void *arg) {
    (void)arg;
    while (keep_running) {
        int new_fd = accept(tx_listen_fd, NULL, NULL);
        if (new_fd < 0) {
            if (keep_running) perror("accept tx");
            break;
        }
        pthread_mutex_lock(&conn_mutex);
        if (current_tx_fd >= 0) {
            printf("已有 tx 连接，拒绝新连接\n");
            close(new_fd);
        } else {
            current_tx_fd = new_fd;
            printf("tx 已连接 (fd=%d)\n", new_fd);
            pthread_cond_signal(&tx_connected);
        }
        pthread_mutex_unlock(&conn_mutex);
    }
    return NULL;
}

static void *rx_acceptor(void *arg) {
    (void)arg;
    while (keep_running) {
        int new_fd = accept(rx_listen_fd, NULL, NULL);
        if (new_fd < 0) {
            if (keep_running) perror("accept rx");
            break;
        }
        pthread_mutex_lock(&conn_mutex);
        if (current_rx_fd >= 0) {
            printf("已有 rx 连接，拒绝新连接\n");
            close(new_fd);
        } else {
            current_rx_fd = new_fd;
            printf("rx 已连接 (fd=%d)\n", new_fd);
        }
        pthread_mutex_unlock(&conn_mutex);
    }
    return NULL;
}

// ==================== 转发线程 ====================
static void *forwarder(void *arg) {
    (void)arg;
    unsigned int seed = (unsigned int)time(NULL);
    unsigned long frame_count = 0;

    while (keep_running) {
        // ----- 等待 tx 连接 -----
        pthread_mutex_lock(&conn_mutex);
        while (keep_running && current_tx_fd < 0) {
            pthread_cond_wait(&tx_connected, &conn_mutex);
        }
        if (!keep_running) {
            pthread_mutex_unlock(&conn_mutex);
            break;
        }
        int tx_fd = current_tx_fd;
        pthread_mutex_unlock(&conn_mutex);

        printf("转发线程：开始处理 tx (fd=%d) 的数据\n", tx_fd);

        // ----- 处理 tx 数据 -----
        while (keep_running) {
            // 读取 8 字节原始码字
            uint8_t cw_bytes[BCH_BYTES];
            ssize_t n = recv(tx_fd, cw_bytes, BCH_BYTES, MSG_WAITALL);
            if (n <= 0) {
                if (n == 0) printf("tx 连接关闭\n");
                else if (keep_running) perror("recv from tx");
                break;
            }

            // 解包为 63 比特
            int cw_bits[BCH_N];
            bytes_to_bits(cw_bytes, cw_bits);

            // AWGN 信道得到 LLR
            BchFrame frame;
            memcpy(frame.codeword, cw_bytes, BCH_BYTES);
            awgn_channel(cw_bits, frame.llr, args.snr_db, &seed);

            // 检查 rx 状态
            int rx_fd;
            pthread_mutex_lock(&conn_mutex);
            rx_fd = current_rx_fd;
            pthread_mutex_unlock(&conn_mutex);

            if (rx_fd >= 0) {
                // rx 在线：发送 BchFrame
                if (send(rx_fd, &frame, sizeof(BchFrame), 0) < 0) {
                    perror("send to rx");
                    pthread_mutex_lock(&conn_mutex);
                    if (current_rx_fd == rx_fd) {
                        close(current_rx_fd);
                        current_rx_fd = -1;
                        printf("rx 连接断开，恢复丢弃模式\n");
                    }
                    pthread_mutex_unlock(&conn_mutex);
                } else {
                    frame_count++;
                    if (frame_count % 1000 == 0)
                        printf("已转发 %lu 帧\n", frame_count);
                }
            }
            // rx 不在线：数据已读取，直接丢弃
        }

        // tx 断开，清理
        pthread_mutex_lock(&conn_mutex);
        if (current_tx_fd == tx_fd) {
            close(current_tx_fd);
            current_tx_fd = -1;
        }
        pthread_mutex_unlock(&conn_mutex);
        printf("转发线程：tx 连接已关闭，等待新 tx 连接...\n");
    }

    printf("转发线程退出，共转发 %lu 帧\n", frame_count);
    return NULL;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[]) {
    parse_arguments(argc, argv);
    signal(SIGINT,  sigint_handler);
    signal(SIGTERM, sigint_handler);

    tx_listen_fd = create_listen_socket(args.tx_port);
    rx_listen_fd = create_listen_socket(args.rx_port);

    pthread_t tx_th, rx_th, fwd_th;
    pthread_create(&tx_th, NULL, tx_acceptor, NULL);
    pthread_create(&rx_th, NULL, rx_acceptor, NULL);
    pthread_create(&fwd_th, NULL, forwarder, NULL);

    printf("信道模拟器启动，SNR = %.1f dB，按 Ctrl+C 退出\n", args.snr_db);

    pthread_join(fwd_th, NULL);
    keep_running = 0;
    pthread_join(tx_th, NULL);
    pthread_join(rx_th, NULL);

    close(tx_listen_fd);
    close(rx_listen_fd);
    printf("信道模拟器已退出\n");
    return 0;
}