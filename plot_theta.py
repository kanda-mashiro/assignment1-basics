import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
d = 128

# 定义 theta 函数
def theta(t):
    return 10000 ** (-2 * t / d)

# 定义 f(m) 函数
def f(m):
    total_sum = 0
    for j in range(d // 2):
        # 计算内层求和: Sum[Exp[I*m*theta[i]], {i, 0, j}]
        inner_sum = 0
        for i in range(j + 1):
            inner_sum += np.exp(1j * m * theta(i))
        # 计算范数（绝对值）
        total_sum += np.abs(inner_sum)
    # 返回平均值
    return total_sum / (d / 2)

def f2(m):
    total_sum = 0
    for j in range(d // 2):
        # 计算内层求和: Sum[Exp[I*m*theta[i]], {i, 0, j}]
        inner_sum = 0
        for i in range(j + 1):
            inner_sum += np.exp(1j * m * theta(i+1))
        # 计算范数（绝对值）
        total_sum += np.abs(inner_sum)
    # 返回平均值
    return total_sum / (d / 2)

# 生成 m 值
m_values = np.linspace(0, 256, 500)

# 计算 f(m) 值
f_values = [f(m) for m in m_values]
f2_values = [f2(m) for m in m_values]

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(m_values, f_values, linewidth=2, color='red')
plt.plot(m_values, f2_values, linewidth=2, color='blue')
plt.xlabel('相对距离', fontsize=12)
plt.ylabel('相对大小', fontsize=12)
plt.title('f(m) vs m', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图形
plt.savefig('theta_plot.png', dpi=300, bbox_inches='tight')
print("图表已保存为 theta_plot.png")

# 显示图形
plt.show()
