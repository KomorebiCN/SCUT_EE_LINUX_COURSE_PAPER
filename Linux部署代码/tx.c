/*
 * tx.c —— 多线程发送端（BCH(63,36) 编码，持续发送模式）
 * 依赖：common.h
 * 用法：./tx <IP> <端口> <线程数>
 * 编译：gcc -std=c99 -Wall -Wextra -pthread -o tx tx.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "common.h"
#include "model.h"

// ==================== BCH(63,36) 生成矩阵 ====================


// BCH(63,36) 编码：msg[36] → cw_bytes[8]（压缩字节形式）
static void bch_encode(const int msg[BCH_K], uint8_t cw_bytes[BCH_BYTES]) {
    int cw[BCH_N];
    memset(cw, 0, sizeof(cw));
    for (int i = 0; i < BCH_K; i++) {
        if (msg[i]) {
            for (int j = 0; j < BCH_N; j++) {
                cw[j] ^= bp_G[i][j];
            }
        }
    }
    bits_to_bytes(cw, cw_bytes);
}

// ==================== 全局变量 ====================
static volatile sig_atomic_t keep_running = 1;
static int shared_sockfd = -1;

// 共享队列（元素为 8 字节压缩码字）
#define QUEUE_SIZE 128
static uint8_t queue[QUEUE_SIZE][BCH_BYTES];
static int q_front = 0, q_count = 0;
static pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  q_not_empty = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  q_not_full  = PTHREAD_COND_INITIALIZER;

// ==================== 信号处理 ====================
void sigint_handler(int sig) {
    (void)sig;
    keep_running = 0;
    pthread_cond_broadcast(&q_not_empty);
    pthread_cond_broadcast(&q_not_full);
}

// ==================== 参数解析 ====================
typedef struct {
    char ip[64];
    int  port;
    int  thread_count;
} Arguments;

static void parse_arguments(int argc, char *argv[], Arguments *args) {
    if (argc != 4) {
        fprintf(stderr, "用法: %s <IP地址> <端口> <线程数>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    strncpy(args->ip, argv[1], sizeof(args->ip) - 1);
    args->ip[sizeof(args->ip) - 1] = '\0';
    args->port = atoi(argv[2]);
    args->thread_count = atoi(argv[3]);
    if (args->port <= 0 || args->port > 65535 || args->thread_count < 1) {
        fprintf(stderr, "参数非法: 端口 %d, 线程数 %d\n", args->port, args->thread_count);
        exit(EXIT_FAILURE);
    }
}

// ==================== TCP 连接 ====================
static int connect_to_channel(const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(EXIT_FAILURE); }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(fd);
        exit(EXIT_FAILURE);
    }

    if (connect(fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect");
        close(fd);
        exit(EXIT_FAILURE);
    }
    return fd;
}

// ==================== 生产者线程 ====================
typedef struct {
    int tid;
} ThreadArg;

static void *producer_thread(void *arg) {
    ThreadArg *targ = (ThreadArg*)arg;
    unsigned int seed = (unsigned int)(time(NULL) + targ->tid);

    while (keep_running) {
        // 1. 生成随机 36 位信息比特
        int msg[BCH_K];
        for (int i = 0; i < BCH_K; i++) {
            msg[i] = rand_r(&seed) % 2;
        }

        // 2. BCH 编码得到 8 字节压缩码字
        uint8_t cw_bytes[BCH_BYTES];
        bch_encode(msg, cw_bytes);

        // 3. 放入共享队列
        pthread_mutex_lock(&q_lock);
        while (q_count == QUEUE_SIZE && keep_running) {
            pthread_cond_wait(&q_not_full, &q_lock);
        }
        if (!keep_running) {
            pthread_mutex_unlock(&q_lock);
            break;
        }

        int idx = (q_front + q_count) % QUEUE_SIZE;
        memcpy(queue[idx], cw_bytes, BCH_BYTES);
        q_count++;
        pthread_cond_signal(&q_not_empty);
        pthread_mutex_unlock(&q_lock);

        usleep(10);
    }
    pthread_exit(NULL);
}

// ==================== 消费者线程 ====================
static void *sender_consumer(void *arg) {
    (void)arg;
    while (keep_running || q_count > 0) {
        pthread_mutex_lock(&q_lock);
        while (q_count == 0 && keep_running) {
            pthread_cond_wait(&q_not_empty, &q_lock);
        }
        if (q_count == 0) {
            pthread_mutex_unlock(&q_lock);
            break;
        }

        uint8_t cw_bytes[BCH_BYTES];
        memcpy(cw_bytes, queue[q_front], BCH_BYTES);
        q_front = (q_front + 1) % QUEUE_SIZE;
        q_count--;
        pthread_cond_signal(&q_not_full);
        pthread_mutex_unlock(&q_lock);

        // 发送 8 字节压缩码字
        if (send(shared_sockfd, cw_bytes, BCH_BYTES, 0) < 0) {
            perror("send");
            keep_running = 0;
            break;
        }
    }
    pthread_exit(NULL);
}

// ==================== 主函数 ====================
int main(int argc, char *argv[]) {
    Arguments args;
    parse_arguments(argc, argv, &args);

    signal(SIGINT,  sigint_handler);
    signal(SIGTERM, sigint_handler);

    shared_sockfd = connect_to_channel(args.ip, args.port);
    printf("已连接到信道模拟器 %s:%d（持续发送模式，按 Ctrl+C 退出）\n",
           args.ip, args.port);

    int n_threads = args.thread_count;
    pthread_t *producers = malloc(sizeof(pthread_t) * n_threads);
    ThreadArg *targs     = malloc(sizeof(ThreadArg) * n_threads);
    if (!producers || !targs) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n_threads; i++) {
        targs[i].tid = i;
        if (pthread_create(&producers[i], NULL, producer_thread, &targs[i]) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }

    pthread_t consumer;
    if (pthread_create(&consumer, NULL, sender_consumer, NULL) != 0) {
        perror("pthread_create");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_join(producers[i], NULL);
    }

    keep_running = 0;
    pthread_cond_broadcast(&q_not_empty);
    pthread_join(consumer, NULL);

    close(shared_sockfd);
    free(producers);
    free(targs);
    printf("发送完成，程序退出。\n");
    return 0;
}