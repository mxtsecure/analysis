import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 设置文件路径
file_path = "/data/xiangtao/projects/crossdefense/code/analysis/results/interpretability/01-identity_critical_para/Llama-3.2-1B-Instruct-tofu/privacy/module_risk_scores.json"
save_path = "/data/xiangtao/projects/crossdefense/code/analysis/results/visualization/01-identity_critical_para/Llama-3.2-1B-Instruct-tofu/privacy"
os.makedirs(save_path, exist_ok=True)

# 2. 读取JSON文件
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
except FileNotFoundError:
    print(f"错误：文件未找到在 {file_path}")
    exit()
except json.JSONDecodeError:
    print("错误：JSON文件格式不正确，无法解析。")
    exit()

try:
    data = full_data['modules'] 
except KeyError:
    print("错误：JSON文件中未找到预期的归因分数键（如 'all_module_scores'）。请检查 JSON 结构并修改代码中的键名。")
    exit()


# 3. 提取归因值（只看 mean 字段）并构建 DataFrame
# 现在 data 变量是：{ "module_name_1": { "mean": [...] }, ... }

# 存储所有模块的均值列表
module_mean_scores = {}
num_layers = 0

# 遍历 JSON 中的所有键（这些键应该是模块名）
for module_key, scores_dict in data.items():
    if 'mean' in scores_dict:
        # scores_dict['mean'] 应该是一个列表的列表，例如 [[0.1], [0.2], ...]
        # 我们从嵌套列表中提取单个分数：[0.1, 0.2, ...]
        mean_scores = [item for item in scores_dict['mean']]
        module_mean_scores[module_key] = mean_scores
        
        # 确定层的数量（假设所有模块的层数都相同）
        if num_layers == 0:
            num_layers = len(mean_scores)

# **【检查】 验证数据是否成功提取**
if not module_mean_scores:
    print("错误：未能从 JSON 中提取任何模块的 'mean' 归因分数。请检查 JSON 结构。")
    exit()

# 转换为 Pandas DataFrame
# DataFrame 的行（索引）是层编号
layer_indices = [f"Layer {i}" for i in range(num_layers)]
df_scores = pd.DataFrame(module_mean_scores, index=layer_indices)


# 4. 绘制热力图

plt.figure(figsize=(12, 10))

# 使用 seaborn 绘制热力图
sns.heatmap(
    df_scores,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r", # 常用色系，红色-蓝色
    linewidths=0.5,
    cbar_kws={'label': 'Mean Attribution Score'}
)

# 设置图表标题和坐标轴标签
plt.title('Layer vs. Module Mean Attribution Scores Heatmap', fontsize=16)
plt.xlabel('Module', fontsize=14)
plt.ylabel('Layer Index', fontsize=14)

# 确保 Y 轴标签（层）以递增顺序显示，并且底部是第一层
plt.gca().invert_yaxis()

plt.tight_layout() # 自动调整布局以防止标签重叠
# 5. 保存图片
output_filename = "layer_module_attribution_heatmap.png"
plt.savefig(os.path.join(save_path, output_filename), dpi=300)
print(f"热力图已保存至: {os.path.join(save_path, output_filename)}")

# plt.show() # 如果在非交互式环境运行，可能不需要这行