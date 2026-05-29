import numpy as np
import torch
import galois
import matplotlib.pyplot as plt
from scipy.special import erfc

bch = galois.BCH(63, 36)
H_np = bch.H.astype(np.int32)
H = np.array(H_np, dtype=np.int32)
G_np = bch.G.astype(np.int32)
G_tensor = torch.tensor(G_np, dtype=torch.float32)

def generate_awgn_llr(codewords, snr_db):
    """
    通过AWGN信道传输，生成LLR
    codewords: 传输的码字 (batch_size, n) 中的值为 {0,1}
    snr_db: 信噪比(dB)
    返回: LLR (batch_size, n)
    """
    batch_size, n = codewords.shape
    # 将0,1转换为BPSK符号: 0->+1, 1->-1
    symbols = 1 - 2 * codewords  # 0->1, 1->-1

    # 计算噪声方差 sigma^2 = 1 / (2 * 10^(SNR/10))
    snr_linear = 10 ** (snr_db / 10)
    sigma2 = 1 / (2 * snr_linear)
    sigma = np.sqrt(sigma2)

    # 添加高斯噪声
    noise = torch.randn_like(symbols) * sigma
    received = symbols + noise

    # 计算LLR: log(P(y|x=1)/P(y|x=-1)) = 2*y/sigma^2
    llr = 2 * received / sigma2
    return llr

def compute_initial_LLR(received_signal, noise_variance):
    return 2 * received_signal / noise_variance

def bp_decoder(H, llr, max_iter):
    num_vars = H.shape[1]
    num_checks = H.shape[0]
    # 初始化消息
    var_to_check = np.zeros((num_checks, num_vars))
    check_to_var = np.zeros((num_checks, num_vars))
    for iter in range(max_iter):
        # 校验节点到变量节点的消息更新
        for c in range(num_checks):
            for v in range(num_vars):
                if H[c, v] == 1:
                    product = 1.0
                    for v2 in range(num_vars):
                        if H[c, v2] == 1 and v2 != v:
                            product *= np.tanh(var_to_check[c, v2] / 2)
                    check_to_var[c, v] = 2 * np.arctanh(product)
        # 变量节点到校验节点的消息更新
        for v in range(num_vars):
            for c in range(num_checks):
                if H[c, v] == 1:
                    sum_msg = llr[v]
                    for c2 in range(num_checks):
                        if H[c2, v] == 1 and c2 != c:
                            sum_msg += check_to_var[c2, v]
                    var_to_check[c, v] = sum_msg
        # 收敛判断
        decoded_bits = (np.sum(check_to_var, axis=0) + llr) < 0
        if np.all(np.dot(H, decoded_bits) % 2 == 0):
            break
    return decoded_bits

def generate_random_codeword(batch_size):
    """
    使用生成矩阵 G 生成随机的 BCH(63,36) 码字。
    """
    # 随机信息比特 (batch_size, k)
    msg = torch.randint(0, 2, (batch_size, 36), dtype=torch.float32)
    # 编码： msg * G  mod 2
    codewords = torch.matmul(msg, G_tensor) % 2
    return codewords

# ==================== 测试参数 ====================
snr_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0]  # SNR 测试点 (dB)
num_codewords_per_snr = 2000              # 每个 SNR 测试的码字数
max_bp_iter = 5                           # BP 译码最大迭代次数

# ==================== 测试传统 BP 译码 ====================
ber_bp = []  # 存储每个 SNR 的 BER

for snr_db in snr_list:
    # 生成随机码字
    codewords = generate_random_codeword(num_codewords_per_snr)  # tensor (batch, 63)
    codewords_np = codewords.numpy().astype(np.int32)

    # 通过 AWGN 信道得到 LLR
    llr_tensor = generate_awgn_llr(codewords, snr_db)            # tensor (batch, 63)
    llr_np = llr_tensor.numpy()

    total_bit_errors = 0
    total_bits = 0

    # 逐帧译码
    for i in range(num_codewords_per_snr):
        llr_i = llr_np[i]                   # (63,)
        original_bits = codewords_np[i]     # (63,)

        # 传统 BP 译码
        decoded_bits = bp_decoder(H, llr_i, max_bp_iter)  # 返回 numpy bool 数组

        # 统计误比特
        errors = np.sum(decoded_bits != original_bits)
        total_bit_errors += errors
        total_bits += 63

    ber = total_bit_errors / total_bits
    ber_bp.append(ber)
    print(f"SNR = {snr_db} dB, BP BER = {ber:.6e}")

# ==================== 计算无编码 BPSK 理论 BER ====================
snr_linear = 10 ** (np.array(snr_list) / 10)
ber_uncoded = 0.5 * erfc(np.sqrt(snr_linear))

# ==================== 绘图 ====================
plt.figure(figsize=(7, 5))
plt.semilogy(snr_list, ber_bp, 'ro-', linewidth=1.5, markersize=8,
             label='Traditional BP Decoder')
plt.semilogy(snr_list, ber_uncoded, 's--', color='gray', linewidth=1.2,
             markersize=7, label='Uncoded BPSK (theory)')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('BER', fontsize=13)
plt.grid(True, which='major', linestyle='-', alpha=0.5)
plt.grid(True, which='minor', linestyle='--', alpha=0.25)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig('ber_bp_vs_uncoded.pdf', dpi=300, bbox_inches='tight')
plt.show()