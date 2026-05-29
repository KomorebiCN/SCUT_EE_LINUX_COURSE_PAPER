import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import galois

bch = galois.BCH(63, 36)

# 获取生成矩阵和校验矩阵
G_np = bch.G.astype(np.int32)
H_np = bch.H.astype(np.int32)
H = H_np

n = H.shape[1]
m = H.shape[0]

var_to_check = [[] for _ in range(n)]
check_to_var = [[] for _ in range(m)]

# 枚举H矩阵的所有元素，值为1的地方就是一条边
edges = []  # 存储所有边，格式 (v, c)
for i in range(m):
    for j in range(n):
        if H[i, j] == 1:
            edges.append((j, i))  # (变量节点, 校验节点)
            var_to_check[j].append(i)
            check_to_var[i].append(j)

# 边的数量 (E)
E = len(edges)

print(f"Number of variable nodes: {n}")
print(f"Number of check nodes: {m}")
print(f"Number of edges: {E}")
print("Variable nodes to Check nodes connections:", var_to_check)
print("Check nodes to Variable nodes connections:", check_to_var)

# 为每条边分配一个唯一的索引
edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}


# ------------------------------
# 构建深度神经网络解码器 (Weighted BP Decoder)
# ------------------------------
class WeightedBPDecoder(nn.Module):

    def __init__(self, n, m, edges, var_to_check, check_to_var, edge_to_idx, num_iterations=5, use_multiloss=False):
        super(WeightedBPDecoder, self).__init__()

        self.n = n
        self.m = m
        self.E = len(edges)
        self.edges = edges
        self.var_to_check = var_to_check
        self.check_to_var = check_to_var
        self.edge_to_idx = edge_to_idx
        self.num_iterations = num_iterations
        self.use_multiloss = use_multiloss

        #这几个权重，只有设置乘的系数为这么大的时候才会练出来效果比较好

        # ---------- 输入层权重 w_{i,v} ----------
        self.w_input = nn.Parameter(torch.randn(num_iterations, n) * 0.05)

        # ---------- 边对权重 w_{i,e,e'} ----------

        # 构建边对列表
        self.edge_pairs = []  # 每个元素是 (e_idx, e_prime_idx)
        # 同时为了方便前向传播，为每个目标边 e 记录其关联的 (e_prime_idx, weight_idx) 列表
        self.target_edge_to_incoming = [[] for _ in range(self.E)]  # 长度 E，每个元素是 [(e_prime_idx, pair_idx), ...]

        for v in range(n):
            neighbors = var_to_check[v]  # 该变量节点连接的所有校验节点 c
            d = len(neighbors)
            if d <= 1:
                continue
            # 对于每一条出边 e = (v, c)
            for idx_c, c in enumerate(neighbors):
                e = (v, c)
                e_idx = edge_to_idx[e]
                # 对于每一个其他传入边 e' = (v, c')，c' != c
                for idx_c2, c2 in enumerate(neighbors):
                    if c2 == c:
                        continue
                    e_prime = (v, c2)
                    e_prime_idx = edge_to_idx[e_prime]
                    pair_idx = len(self.edge_pairs)
                    self.edge_pairs.append((e_idx, e_prime_idx))
                    self.target_edge_to_incoming[e_idx].append((e_prime_idx, pair_idx))

        self.total_pairs = len(self.edge_pairs)
        # 边对权重，形状 (num_iterations, total_pairs)
        self.edge_pair_weights = nn.Parameter(torch.randn(num_iterations, self.total_pairs) * 0.05)

        # # 每条边在每个奇数层有一个独立的权重
        # self.edge_weights = nn.Parameter(torch.ones(num_iterations, E) * 0.1)

        # ---------- 输出层权重 ----------
        # w_{2L+1, v} 和 w_{2L+1, v, e'}
        self.w_out_node = nn.Parameter(torch.randn(n) * 0.05)  # 每个变量节点的标量
        self.w_out_edge = nn.Parameter(torch.randn(n, self.E) * 0.05)  # 每个变量节点和其关联边的权重

        self.clip_val = 10.0

    def forward(self, llr, return_all_outputs=False):
        batch_size = llr.shape[0]
        device = llr.device

        # 初始化 CN->VN 消息为0
        msg_c2v_prev = torch.zeros(batch_size, self.E, device=device)

        if self.use_multiloss and return_all_outputs:
            all_intermediate_outputs = []

        for iter_idx in range(self.num_iterations):
            # ---------- 奇数层: VN -> CN (公式4) ----------
            msg_v2c_curr = torch.zeros(batch_size, self.E, device=device)

            # 对于每个变量节点 v
            for v in range(self.n):
                neighbors = self.var_to_check[v]
                if not neighbors:
                    continue
                # 获取该层该变量节点的输入权重 w_{i,v}
                w_v = self.w_input[iter_idx, v]
                llr_v = llr[:, v].unsqueeze(1)  # (batch,1)

                # 对于每条出边 e = (v,c)
                for c in neighbors:
                    e = (v, c)
                    e_idx = self.edge_to_idx[e]

                    # 收集来自其他边 e' = (v,c') 的消息，并乘以对应的边对权重
                    # 根据 target_edge_to_incoming[e_idx] 获取 (e_prime_idx, pair_idx) 列表
                    weighted_sum = torch.zeros(batch_size, 1, device=device)
                    for (e_prime_idx, pair_idx) in self.target_edge_to_incoming[e_idx]:
                        w_ep = self.edge_pair_weights[iter_idx, pair_idx]  # 权重标量
                        msg_from_e_prime = msg_c2v_prev[:, e_prime_idx].unsqueeze(1)  # (batch,1)
                        weighted_sum = weighted_sum + w_ep * msg_from_e_prime

                    # # 直接求和（不加权），然后整体乘以该边的权重
                    # sum_others = torch.zeros(batch_size, 1, device=device)
                    # for c2 in neighbors:
                    #     if c2 == c: continue
                    #     e2 = (v, c2)
                    #     e2_idx = self.edge_to_idx[e2]
                    #     sum_others += msg_c2v_prev[:, e2_idx].unsqueeze(1)
                    # # 该边的权重（第 iter 层，第 e_idx 条边）
                    # w_e = self.edge_weights[iter_idx, e_idx]
                    # weighted_sum = w_e * sum_others


                    # 公式4: tanh(0.5 * (w_v * llr_v + weighted_sum))
                    input_tanh = w_v * llr_v + weighted_sum
                    input_tanh = torch.clamp(input_tanh, -self.clip_val, self.clip_val)
                    msg_v2c_curr[:, e_idx] = torch.tanh(0.5 * input_tanh).squeeze(1)

            # ---------- 偶数层: CN -> VN (公式5, 无权重) ----------
            msg_c2v_curr = torch.zeros(batch_size, self.E, device=device)
            for c in range(self.m):
                connected_v = self.check_to_var[c]
                if not connected_v:
                    continue
                for v in connected_v:
                    e = (v, c)
                    e_idx = self.edge_to_idx[e]
                    # 计算乘积 prod_{v' != v} tanh(msg_v2c_curr/2)
                    prod = torch.ones(batch_size, 1, device=device)
                    for v_prime in connected_v:
                        if v_prime == v:
                            continue
                        e_prime = (v_prime, c)
                        e_prime_idx = self.edge_to_idx[e_prime]
                        # 注意：使用当前奇数层刚计算出的 msg_v2c_curr
                        # prod = prod * torch.tanh(0.5 * msg_v2c_curr[:, e_prime_idx].unsqueeze(1))
                        prod = prod * msg_v2c_curr[:, e_prime_idx].unsqueeze(1)
                    # 避免 atanh 输入为 ±1
                    prod = torch.clamp(prod, -0.9999, 0.9999)
                    msg_c2v_curr[:, e_idx] = 2 * torch.atanh(prod).squeeze(1)

            msg_c2v_prev = msg_c2v_curr

            # ---- 如果使用 multi-loss，收集中间输出 ----
            if self.use_multiloss and return_all_outputs:
                # 计算当前奇数层后的输出 (公式6)
                output = torch.zeros(batch_size, self.n, device=device)
                for v in range(self.n):
                    out_val = self.w_out_node[v] * llr[:, v]
                    for c in self.var_to_check[v]:
                        e = (v, c)
                        e_idx = self.edge_to_idx[e]
                        out_val = out_val + self.w_out_edge[v, e_idx] * msg_c2v_curr[:, e_idx]
                    output[:, v] = out_val
                output_prob = torch.sigmoid(output)
                all_intermediate_outputs.append(output_prob)

        # ---------- 最终输出层 ----------
        final_output = torch.zeros(batch_size, self.n, device=device)
        for v in range(self.n):
            out_val = self.w_out_node[v] * llr[:, v]
            for c in self.var_to_check[v]:
                e = (v, c)
                e_idx = self.edge_to_idx[e]
                out_val = out_val + self.w_out_edge[v, e_idx] * msg_c2v_prev[:, e_idx]
            final_output[:, v] = out_val
        final_output_prob = torch.sigmoid(final_output)

        if self.use_multiloss and return_all_outputs:
            all_intermediate_outputs.append(final_output_prob)
            return all_intermediate_outputs
        else:
            return final_output_prob


# ------------------------------
#  辅助函数: 数据生成和训练
# ------------------------------
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


def bpsk_modulation(codewords):
    """将二进制码字转换为BPSK符号 (0->1, 1->-1)"""
    return 1 - 2 * codewords


def binary_cross_entropy_loss(outputs, targets):
    """标准交叉熵损失，假设outputs是概率"""
    # targets是0/1，形状 (batch_size, n)
    eps = 1e-8
    outputs = torch.clamp(outputs, eps, 1 - eps)
    loss = - (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
    return loss.mean()


def train_one_epoch(model, optimizer, train_loader, device, snr_range, use_multiloss=False):
    # model.train()
    total_loss = 0
    total_batches = len(train_loader)
    for i, batch_codewords in enumerate(train_loader):
        batch_codewords = batch_codewords.to(device)
        batch_size = batch_codewords.shape[0]

        # 随机选择SNR
        snr_db = np.random.uniform(snr_range[0], snr_range[1])

        # 生成LLR
        llr = generate_awgn_llr(batch_codewords, snr_db)

        optimizer.zero_grad()

        if use_multiloss:
            # 使用Multi-loss: 所有奇数层输出的加权和
            all_outputs = model(llr, return_all_outputs=True)
            # 计算所有输出的平均损失
            loss = torch.tensor(0.0, device=device)
            for output in all_outputs:
                loss = loss + binary_cross_entropy_loss(output, batch_codewords)
            loss = loss / len(all_outputs)
        else:
            outputs = model(llr, return_all_outputs=False)
            loss = binary_cross_entropy_loss(outputs, batch_codewords)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def test_model(model, test_loader, device, snr_db_list):
    """测试模型在不同SNR下的误比特率(BER)"""
    model.eval()
    ber_results = []

    for snr_db in snr_db_list:
        total_bits = 0
        error_bits = 0

        with torch.no_grad():
            for batch_codewords in test_loader:
                batch_codewords = batch_codewords.to(device)
                batch_size = batch_codewords.shape[0]

                # 生成LLR
                llr = generate_awgn_llr(batch_codewords, snr_db)

                # 解码
                # outputs = model(llr, return_all_outputs=False)
                outputs = model(llr)

                # 硬判决: 概率 > 0.5 判为1
                decoded = (outputs > 0.5).float()

                # 统计错误比特数
                errors = (decoded != batch_codewords).sum().item()
                total_bits += batch_codewords.numel()
                error_bits += errors

        ber = error_bits / total_bits
        ber_results.append(ber)
        print(f"SNR = {snr_db} dB, BER = {ber:.6f}")

    return ber_results


def export_to_c_header(model, filename, n, m, E, k, num_iterations, total_pairs,
                       var_to_check, check_to_var, target_edge_to_incoming,
                       G_np, H_np):
    """
    将模型权重、Tanner 图结构以及生成矩阵 G 和校验矩阵 H 导出为 C 头文件。
    所有权重和矩阵都以多维数组形式存储，方便 C 代码直接索引。
    """
    with open(filename, 'w') as f:
        f.write("// Auto-generated from trained PyTorch model\n")
        f.write("#ifndef BP_DECODER_WEIGHTS_H\n")
        f.write("#define BP_DECODER_WEIGHTS_H\n\n")

        # ------------------------------
        # 1. 基本参数
        # ------------------------------
        f.write(f"#define BP_N {n}\n")
        f.write(f"#define BP_M {m}\n")
        f.write(f"#define BP_K {k}\n")
        f.write(f"#define BP_E {E}\n")
        f.write(f"#define BP_NUM_ITERATIONS {num_iterations}\n")
        f.write(f"#define BP_TOTAL_PAIRS {total_pairs}\n\n")

        # 计算最大度数（变量节点和校验节点的最大邻居数）
        max_var_degree = max(len(lst) for lst in var_to_check) if var_to_check else 0
        max_check_degree = max(len(lst) for lst in check_to_var) if check_to_var else 0
        max_degree = max(max_var_degree, max_check_degree)
        f.write(f"#define BP_MAX_DEGREE {max_degree}\n\n")

        # 构建 bp_edge_idx 二维数组 (n x m)，默认 -1
        edge_idx_map = [[-1] * m for _ in range(n)]
        for (v, c), idx in edge_to_idx.items():
            edge_idx_map[v][c] = idx

        f.write(f"const int bp_edge_idx[{n}][{m}] = {{\n")
        for v in range(n):
            f.write("    {")
            row = edge_idx_map[v]
            f.write(', '.join(str(x) for x in row))
            f.write("}")
            if v < n - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # ------------------------------
        # 2. 生成矩阵 G 和校验矩阵 H (整数 0/1)
        # ------------------------------
        # G: (k, n)
        f.write(f"const int bp_G[{k}][{n}] = {{\n")
        for i in range(k):
            f.write("    {")
            for j in range(n):
                f.write(f"{G_np[i, j]}")
                if j < n - 1:
                    f.write(", ")
            f.write("}")
            if i < k - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # H: (m, n)
        f.write(f"const int bp_H[{m}][{n}] = {{\n")
        for i in range(m):
            f.write("    {")
            for j in range(n):
                f.write(f"{H_np[i, j]}")
                if j < n - 1:
                    f.write(", ")
            f.write("}")
            if i < m - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # ------------------------------
        # 3. 权重数组 (float) 保持原始维度
        # ------------------------------
        state_dict = model.state_dict()

        # w_input: (num_iterations, n)
        w_input = state_dict['w_input'].detach().cpu().numpy()
        f.write(f"const float bp_w_input[{num_iterations}][{n}] = {{\n")
        for i in range(num_iterations):
            f.write("    {")
            for j in range(n):
                f.write(f"{w_input[i, j]:.15f}")
                if j < n - 1:
                    f.write(", ")
            f.write("}")
            if i < num_iterations - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # edge_pair_weights: (num_iterations, total_pairs)
        edge_pair_weights = state_dict['edge_pair_weights'].detach().cpu().numpy()
        f.write(f"const float bp_edge_pair_weights[{num_iterations}][{total_pairs}] = {{\n")
        for i in range(num_iterations):
            f.write("    {")
            for j in range(total_pairs):
                f.write(f"{edge_pair_weights[i, j]:.15f}")
                if j < total_pairs - 1:
                    f.write(", ")
            f.write("}")
            if i < num_iterations - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # w_out_node: (n,)
        w_out_node = state_dict['w_out_node'].detach().cpu().numpy()
        f.write(f"const float bp_w_out_node[{n}] = {{")
        for j in range(n):
            f.write(f"{w_out_node[j]:.15f}")
            if j < n - 1:
                f.write(", ")
        f.write("};\n\n")

        # w_out_edge: (n, E)
        w_out_edge = state_dict['w_out_edge'].detach().cpu().numpy()
        f.write(f"const float bp_w_out_edge[{n}][{E}] = {{\n")
        for i in range(n):
            f.write("    {")
            for j in range(E):
                f.write(f"{w_out_edge[i, j]:.15f}")
                if j < E - 1:
                    f.write(", ")
            f.write("}")
            if i < n - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # ------------------------------
        # 4. Tanner 图连接结构 (保持偏移量方式，因为长度不规则)
        # ------------------------------
        # var_to_check 数据与偏移量
        var_check_data = []
        var_check_offsets = [0]
        for v in range(n):
            lst = var_to_check[v]
            var_check_data.extend(lst)
            var_check_offsets.append(len(var_check_data))
        f.write(f"const int bp_var_check_data[] = {{ {', '.join(map(str, var_check_data))} }};\n")
        f.write(f"const int bp_var_check_offsets[] = {{ {', '.join(map(str, var_check_offsets))} }};\n\n")

        # check_to_var 数据与偏移量
        check_var_data = []
        check_var_offsets = [0]
        for c in range(m):
            lst = check_to_var[c]
            check_var_data.extend(lst)
            check_var_offsets.append(len(check_var_data))
        f.write(f"const int bp_check_var_data[] = {{ {', '.join(map(str, check_var_data))} }};\n")
        f.write(f"const int bp_check_var_offsets[] = {{ {', '.join(map(str, check_var_offsets))} }};\n\n")

        # target_edge_to_incoming: 每个目标边的传入边列表
        inc_data = []
        inc_offsets = [0]
        for e_idx in range(E):
            inc_list = target_edge_to_incoming[e_idx]
            for (ep, pair) in inc_list:
                inc_data.append(ep)
                inc_data.append(pair)
            inc_offsets.append(len(inc_data) // 2)
        f.write(f"const int bp_inc_data[] = {{ {', '.join(map(str, inc_data))} }};\n")
        f.write(f"const int bp_inc_offsets[] = {{ {', '.join(map(str, inc_offsets))} }};\n\n")

        f.write("#endif // BP_DECODER_WEIGHTS_H\n")


def generate_golden_test_header(model, n, device, filename="golden_test.h"):
    """
    生成包含单个固定输入 LLR 和对应模型输出概率的 C 头文件。
    """

    # 生成单个固定输入 LLR (n,)
    llr_input = torch.randn(n, device=device) * 1.5

    model.eval()
    with torch.no_grad():
        # 需要增加 batch 维度，模型要求输入 (batch, n)
        output_prob = model(llr_input.unsqueeze(0), return_all_outputs=False)  # (1, n)
        output_prob = output_prob.squeeze(0)  # (n,)

    llr_np = llr_input.cpu().numpy()
    prob_np = output_prob.cpu().numpy()

    with open(filename, 'w') as f:
        f.write("// Auto-generated golden test data (single sample)\n")
        f.write("#ifndef GOLDEN_TEST_H\n")
        f.write("#define GOLDEN_TEST_H\n\n")
        f.write(f"#define GOLDEN_N {n}\n\n")

        # 输入 LLR 数组
        f.write("static const float golden_llr[GOLDEN_N] = {\n    ")
        f.write(', '.join(f"{x:.8f}" for x in llr_np))
        f.write("\n};\n\n")

        # 期望输出概率数组
        f.write("static const float golden_output[GOLDEN_N] = {\n    ")
        f.write(', '.join(f"{x:.8f}" for x in prob_np))
        f.write("\n};\n\n")

        f.write("#endif // GOLDEN_TEST_H\n")

    print(f"Golden test header saved to {filename}")

def set_global_seed(seed=42):
    """一键设置所有需要的随机种子，保证实验可复现性"""
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # 为当前GPU设置种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    # 以下设置可让CUDA算法完全确定性，但会略微降低性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ------------------------------
#  主程序: 训练和测试
# ------------------------------
def main():
    # 超参数设置
    batch_size = 120  # 小批量大小，与论文一致
    num_iterations = 5  # BP迭代次数 (L=5)
    learning_rate = 0.1
    num_epochs = 60
    snr_range = (1, 1)  # 训练SNR范围 (dB)
    test_snrs = [1, 2,3,4,5,6]  # 测试SNR点
    use_multiloss = True  # 是否使用Multi-loss训练
    # set_global_seed(44442)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型
    model = WeightedBPDecoder(
        n=n, m=m, edges=edges,
        var_to_check=var_to_check,
        check_to_var=check_to_var,
        edge_to_idx=edge_to_idx,
        num_iterations=num_iterations,
        use_multiloss=use_multiloss
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 优化器: 使用RMSprop
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 生成训练数据和测试数据
    # 由于零码字是所有线性码都包含的，我们只需要训练零码字
    # 为了方便，生成一个全是0的码字数据集
    train_data = torch.zeros(batch_size, n)  # 全零码字
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    G_tensor = torch.tensor(G_np, dtype=torch.float32)
    # 测试数据: 我们需要测试所有可能的码字，但零码字训练后，模型对所有码字应具有相同的性能
    def generate_random_codeword(batch_size):
        """
        使用生成矩阵 G 生成随机的 BCH(63,36) 码字。
        """
        # 随机信息比特 (batch_size, k)
        msg = torch.randint(0, 2, (batch_size, 36), dtype=torch.float32)
        # 编码： msg * G  mod 2
        codewords = torch.matmul(msg, G_tensor) % 2
        return codewords

    # 创建测试数据加载器
    class RandomCodeDataset(torch.utils.data.Dataset):
        def __init__(self, size, n):
            self.size = size
            self.n = n

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # 生成随机码字
            return generate_random_codeword(1).squeeze(0)

    test_dataset = RandomCodeDataset(500, n)  # 200个随机码字用于测试
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    # 训练循环
    print("Starting training...")
    train_losses = []
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, snr_range, use_multiloss)
        scheduler.step()
        train_losses.append(loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}, LR: {current_lr:.6f}")

    # 绘制训练损失曲线
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    print("\nTesting Weighted BP Decoder...")
    ber_weighted = test_model(model, test_loader, device, test_snrs)

    # 绘制 BER 曲线
    plt.figure()
    plt.semilogy(test_snrs, ber_weighted, 'o-', label='DeepNeural BP')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER Performance')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('ber_comparison.png', dpi=300, bbox_inches='tight')  # 先保存
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'weighted_bp_decoder_bch(63,36).pth')
    export_to_c_header(
        model=model,
        filename="model.h",
        n=model.n,
        m=model.m,
        k=36,
        E=model.E,
        num_iterations=model.num_iterations,
        total_pairs=model.total_pairs,
        var_to_check=model.var_to_check,
        check_to_var=model.check_to_var,
        target_edge_to_incoming=model.target_edge_to_incoming,
        G_np=G_np,
        H_np=H_np
    )
    generate_golden_test_header(model, n, device, filename="golden_test.h")
    print("Model saved to 'model.h'")


if __name__ == "__main__":
    main()