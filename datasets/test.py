import json
import random
import os

def construct_fairness_datasets(file_path, output_limit=100):
    """
    从 StereoSet 构造 Risk (D_R) 和 Normal (D_N) 数据集
    格式: text = context + " " + sentence
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 容器
    D_R = [] # Risk: Stereotype
    D_N = [] # Normal: Anti-stereotype
    
    # 遍历 intrasentence 数据
    for cluster in data.get('data', {}).get('intrasentence', []):
        
        # 1. 获取 Context
        # 例如: "The chess player was BLANK."
        context_str = cluster.get('context', '').strip()
        
        stereo_sent = None
        anti_stereo_sent = None
        
        # 2. 获取对应的 Sentences
        for sent_obj in cluster['sentences']:
            if sent_obj['gold_label'] == 'stereotype':
                stereo_sent = sent_obj['sentence']
            elif sent_obj['gold_label'] == 'anti-stereotype':
                anti_stereo_sent = sent_obj['sentence']
        
        # 3. 构造 "Context + Sentence" 的组合 Query
        # 只有当成对数据都存在时才添加，保证 D_R 和 D_N 的 context 分布一致
        if stereo_sent and anti_stereo_sent:
            
            # 拼接逻辑：Context + 空格 + Sentence
            # Risk 样本: "The chess player was BLANK. The chess player was Asian."
            risk_text = f"{context_str} {stereo_sent}"
            D_R.append(risk_text)

            # Normal 样本: "The chess player was BLANK. The chess player was hispanic."
            benign_text = f"{context_str} {anti_stereo_sent}"
            D_N.append(benign_text)
            
    # 检查数据量
    if len(D_R) < output_limit:
        print(f"警告: 可用数据不足 {output_limit} 条，实际获取 {len(D_R)} 条。")
        output_limit = len(D_R)

    # 4. 打乱顺序 (Shuffle)
    # 此时 D_R 和 D_N 已经是完整的字符串列表，直接打乱即可
    random.seed(42) 
    random.shuffle(D_R)
    random.shuffle(D_N)
    
    # 截取
    D_R = D_R[:output_limit]
    D_N = D_N[:output_limit]
    
    return D_R, D_N

def save_list_to_jsonl(data_list, output_file):
    """
    将字符串列表保存为 jsonl 格式： {"text": "combined_string"}
    """
    print(f"正在写入 {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text_content in data_list:
            # 构造最终的 json 对象
            record = {"text": text_content}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"写入完成，共 {len(data_list)} 行。")

if __name__ == "__main__":
    input_path = "/data/xiangtao/projects/crossdefense/code/defense/fairness/BiasUnlearn/StereoSet/dev.json"
    
    try:
        # 1. 构造数据
        risk_data, benign_data = construct_fairness_datasets(input_path, output_limit=100)
        
        # 2. 定义输出路径
        risk_output_path = "fairness_risk.jsonl"
        benign_output_path = "fairness_benign.jsonl"
        
        # 3. 保存
        save_list_to_jsonl(risk_data, risk_output_path)
        save_list_to_jsonl(benign_data, benign_output_path)
        
        # 4. 预览结果
        print("\n--- Output Preview (fairness_risk.jsonl) ---")
        if os.path.exists(risk_output_path):
            with open(risk_output_path, 'r', encoding='utf-8') as f:
                print(f.readline().strip())
                print(f.readline().strip())
                
    except Exception as e:
        print(f"发生错误: {e}")