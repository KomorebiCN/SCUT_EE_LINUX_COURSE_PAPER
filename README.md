# 项目文件清单

## 目录结构
``` 
./
├── Linux部署代码/ # 在 Linux 上的分布式平台
│   ├── tx.c # 发送端
│   ├── channel.c # 模拟信道
│   ├── rx.c # 接收端
│   ├── common.h # 公共头文件
│   ├── model.h # 训练好的网络结构与权重
│   ├── golden_test.h # Golden Test 
│   ├── CMakeLists.txt
│   ├── model.rar # 这一个模型是多次训练中效果比较好的一次模型 特意归档保存下来 
│   └── build/ # 编译生成的可执行文件与中间文件
│
└── 离线训练与测试代码/ # Python 训练及单文件测试
    ├── 训练与绘图/
    │   ├── train.py # 训练脚本：构建深度展开网络、训练、导出权重
    │   ├── bp_decoder.py # 传统 BP 译码器的python实现，跑起来很慢
    │   ├── plot.py # BER 曲线绘制脚本
    │   ├── requirements.txt 
    │   └── ber_curve.pdf # 最终绘制的 BER 对比图
    │
    └── 部署与测试 on Win/ # Windows 平台测试验证 省的测试的时候来回切系统
        ├── main.c # 单文件整合了译码推理与测试
        ├── model.h 
        ├── golden_test.h 
        └── CMakeLists.txt
```

## 编译方式

### 使用 CMake 构建Linux的部署代码

``` bash
# 1. 创建并进入构建目录 可以换成自己训练出来的model.h和golden_test.h
mkdir build
cd build
# 2. 生成 Makefile
cmake ..
# 3. 编译项目
make
```