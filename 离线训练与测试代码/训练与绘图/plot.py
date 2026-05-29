import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

# ==================== 实测数据 ====================
snr_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9, 10]          # SNR 测试点 (dB)
# ber_measured = [5.260794e-02,2.754127e-02,9.750794e-03,2.055556e-03,2.666667e-04,
#                 4.285714e-05,1.428571e-05,0.000000e+00,0.000000e+00,0.000000e+00]      # 深度展开译码器实测 BER 这一批测试的数据少了点

ber_measured = [5.228857e-02,2.772714e-02,9.683175e-03,2.024127e-03,2.865079e-04,
                3.666667e-05,5.396825e-06,6.349206e-07,4.761905e-07,0.000000e+00]
# ==================================================================

# 无编码 BPSK 理论 BER: Q(sqrt(2 * SNR_linear))
snr_linear = 10 ** (np.array(snr_list) / 10)
ber_uncoded = 0.5 * erfc(np.sqrt(snr_linear))

plt.figure(figsize=(7, 5))

# 实测曲线
plt.semilogy(snr_list, ber_measured, 'bo-', linewidth=1.5,
             markersize=8, label='Weighted BP Decoder (measured)')

# 理论对比曲线
plt.semilogy(snr_list, ber_uncoded, 's--', color='gray', linewidth=1.2,
             markersize=7, label='Uncoded BPSK (theory)')

plt.xlabel('SNR (dB)', fontsize=13)
plt.ylabel('BER', fontsize=13)
plt.grid(True, which='major', linestyle='-', alpha=0.5)
plt.grid(True, which='minor', linestyle='--', alpha=0.25)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig('ber_curve.pdf', dpi=300, bbox_inches='tight')
plt.show()