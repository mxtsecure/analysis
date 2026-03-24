import os
import json
import sys

# 指定配置文件的路径
CONFIG_PATH = "/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/src/configs/safety_defense_pairs.json"

def load_experiments(path):
    """从 JSON 文件加载实验配置"""
    if not os.path.exists(path):
        print(f"❌ Error: Config file not found at {path}")
        sys.exit(1)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ Successfully loaded config from {path}")
            return data
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

def check_experiment_paths():
    # 1. 加载配置
    experiments_data = load_experiments(CONFIG_PATH)
    
    print(f"Checking {len(experiments_data)} experiments...")
    
    missing_paths = []
    
    # 2. 遍历检查
    for i, exp in enumerate(experiments_data):
        exp_id = exp.get("id", f"exp_{i+1}")
        
        # 定义需要检查的三个路径及其标签
        paths_to_check = [
            ("base", exp.get("base")),
            ("defense1", exp.get("defense1")),
            ("defense2", exp.get("defense2"))
        ]
        
        for label, path in paths_to_check:
            if not path:
                print(f"[!] Experiment {exp_id}: Missing path string for '{label}'")
                missing_paths.append((exp_id, label, "Empty Path String"))
                continue
                
            # 检查路径是否存在
            if not os.path.exists(path):
                print(f"[X] Experiment {exp_id}: Path NOT FOUND for '{label}' -> {path}")
                missing_paths.append((exp_id, label, path))
            else:
                # 可选：检查是否为空目录 (仅针对目录)
                if os.path.isdir(path) and not os.listdir(path):
                     print(f"[!] Experiment {exp_id}: Empty Directory for '{label}' -> {path}")

    # 3. 输出总结
    print("\n" + "="*50)
    print("Verification Summary")
    print("="*50)
    
    if not missing_paths:
        print("✅ SUCCESS: All paths exist on the filesystem.")
    else:
        print(f"❌ FAIL: Found {len(missing_paths)} missing paths.")
        for exp_id, label, path in missing_paths:
            print(f"  - {exp_id} [{label}]: {path}")

if __name__ == "__main__":
    check_experiment_paths()