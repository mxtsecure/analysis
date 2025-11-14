import pandas as pd

def csv_to_jsonl_single_text_column(csv_filepath, jsonl_filepath):
    """
    读取每行只有一个文本的 CSV 文件（无列头），并将其转换为以 'text' 为键的 jsonl 文件。

    Args:
        csv_filepath (str): 输入 CSV 文件路径。
        jsonl_filepath (str): 输出 JSONL 文件路径。
    """
    try:
        # 1. 读取 CSV 文件
        df = pd.read_csv(
            csv_filepath, 
            header=None, 
            names=['text'], 
            usecols=[0], # 假设文本在第一列 (索引 0)
            encoding='utf-8', 
            # 这里的 sep 默认是 ','，如果你的文件是纯文本每行一个，
            # 并且确实没有分隔符，可以不指定或指定 sep=None
        )
        
        # 2. 转换为 JSONL (JSON Lines) 格式
        # orient='records'：将每行转换为一个字典 {'text': value}
        # lines=True：确保每条记录占一行，符合 jsonl 格式
        df.to_json(
            jsonl_filepath, 
            orient='records', 
            lines=True, 
            force_ascii=False # 保持中文不被转义
        )
        
        print(f"成功将 '{csv_filepath}' (单文本列) 转换为 '{jsonl_filepath}'。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_filepath}'。")
    except Exception as e:
        print(f"发生错误: {e}")

# --- 示例用法 ---
csv_file = '/data/xiangtao/projects/crossdefense/code/analysis_old/interpretability/03-identity_ciritcal_layer/Safety-Layers/Code/Cos_sim_analysis/retain.csv'
jsonl_file = '/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/retain.jsonl'

csv_to_jsonl_single_text_column(csv_file, jsonl_file)