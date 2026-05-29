
project(DistributedCommSim C)


# 如果需要链接数学库，打开下面这行
link_libraries(m)

# 发送端
add_executable(tx tx.c)

# 信道模拟器
add_executable(channel channel.c)

# AI 接收端
add_executable(rx rx.c)