import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# 保证可以 import analysis.concept_vectors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.concept_vectors import ConceptVector  # 你之前 overlap 脚本里的同一个类


########################################
# 1. 一些辅助函数：参数更新方向
########################################

def get_layer_attn_param_vector(model, layer_idx: int, submodule_name: str = "q_proj") -> torch.Tensor:
    """
    从指定层的 self_attn 子模块中取出参数并展平为一个向量。
    默认取 q_proj，你可以改成 k_proj / v_proj / o_proj。
    """
    # 假设是 LLaMA 结构：model.model.layers[i].self_attn.q_proj
    layer = model.model.layers[layer_idx]
    module = getattr(layer.self_attn, submodule_name)

    vecs = [module.weight.detach().flatten().cpu()]
    if module.bias is not None:
        vecs.append(module.bias.detach().flatten().cpu())

    return torch.cat(vecs, dim=0)  # [D]


def get_param_update_direction(
    base_model,
    defense_model,
    layer_idx: int,
    submodule_name: str = "q_proj",
) -> torch.Tensor:
    """
    计算某一层某个子模块的参数更新方向：
    Δθ = θ_defense - θ_base，并归一化得到方向向量。
    """
    base_vec = get_layer_attn_param_vector(base_model, layer_idx, submodule_name)
    defense_vec = get_layer_attn_param_vector(defense_model, layer_idx, submodule_name)

    delta = defense_vec - base_vec  # [D]
    delta_norm = delta.norm(p=2)
    if delta_norm == 0:
        # 参数完全没变，返回零向量
        return torch.zeros_like(delta)
    return delta / delta_norm


########################################
# 2. ConceptVector -> 一维方向向量
########################################

def concept_to_tensor(obj: ConceptVector | torch.Tensor) -> torch.Tensor:
    """
    将 ConceptVector 或 Tensor 转成归一化的一维向量。

    ⚠️ 这里假设 ConceptVector 内部真正的向量存在于属性 `v`，
    如果你自己的实现不一样（比如叫 `vector` / `direction`），
    把对应那一行改一下即可。
    """
    if isinstance(obj, torch.Tensor):
        vec = obj
    else:
        # ======= 根据你的 ConceptVector 实现在这里修改 =======
        if hasattr(obj, "v"):
            vec = obj.v
        elif hasattr(obj, "vector"):
            vec = obj.vector
        else:
            raise TypeError(
                f"Unsupported ConceptVector structure: {type(obj)}. "
                "Please modify `concept_to_tensor()` to use the correct attribute "
                "(e.g., obj.v, obj.vector, obj.direction, ...)."
            )
        # ====================================================

    vec = vec.detach().cpu().reshape(-1)
    norm = vec.norm(p=2)
    if norm == 0:
        return torch.zeros_like(vec)
    return vec / norm


def load_single_concept_direction(concept_path: Path) -> torch.Tensor:
    """
    加载某一层的 ConceptVector（或 Tensor），并转成一维归一化方向向量。
    """
    obj = torch.load(concept_path, map_location="cpu", weights_only=False)
    return concept_to_tensor(obj)


########################################
# 3. 主函数：逐层计算相似度
########################################

def compute_direction_alignment_with_concepts(
    base_model,
    defense_model,
    concepts_root: str | Path,
    concept_filename: str = "v_safety.pt",  # 激活方向对应的文件名，比如 v_safety.pt / v_privacy.pt / v_delta.pt
    submodule_name: str = "q_proj",
    begin_layer: int | None = None,
    end_layer: int | None = None,
) -> Tuple[Dict[int, float], float]:
    """
    计算“防御参数更新方向”与“激活模式变化方向”(ConceptVector) 的对齐程度。

    假设目录结构类似：
        concepts_root/
          model_layers_0/
            v_safety.pt
          model_layers_1/
            v_safety.pt
          ...

    Args:
        base_model: 未防御模型
        defense_model: 加防御后的模型
        concepts_root: 概念向量根目录
        concept_filename: 每层子目录中的 .pt 文件名
        submodule_name: 注意力子模块 ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        begin_layer: 起始层（含）；默认为 0
        end_layer:   结束层（不含）；默认为 num_hidden_layers

    Returns:
        layer_cos_sims: {layer_idx: cos_sim}
        global_cos_sim: 所有层拼接后的整体余弦相似度
    """
    concepts_root = Path(concepts_root)

    if begin_layer is None:
        begin_layer = 0
    if end_layer is None:
        end_layer = defense_model.config.num_hidden_layers

    layer_cos_sims: Dict[int, float] = {}
    all_param_vecs = []
    all_act_vecs = []

    for layer_idx in range(begin_layer, end_layer):
        layer_dir = concepts_root / f"model_layers_{layer_idx}"
        concept_path = layer_dir / concept_filename

        if not concept_path.exists():
            print(f"[Warning] {concept_path} not found, skip this layer.")
            continue

        # (1) 激活变化方向（来自 ConceptVector）
        act_dir = load_single_concept_direction(concept_path)  # [D_act]

        # (2) 参数更新方向
        param_dir = get_param_update_direction(
            base_model, defense_model, layer_idx, submodule_name=submodule_name
        )  # [D_param]

        # (3) 简单的维度对齐：取较短长度部分（更严谨的做法是你在构造两者时就保证维度一致）
        min_dim = min(param_dir.numel(), act_dir.numel())
        if min_dim == 0:
            print(f"[Warning] empty vector at layer {layer_idx}, skip.")
            continue

        p_vec = param_dir[:min_dim]
        a_vec = act_dir[:min_dim]

        cos_sim = F.cosine_similarity(p_vec, a_vec, dim=0).item()
        layer_cos_sims[layer_idx] = cos_sim

        all_param_vecs.append(p_vec)
        all_act_vecs.append(a_vec)

        print(f"Layer {layer_idx}: cosine similarity = {cos_sim:.6f}")

    # (4) 全局拼接后的整体相似度
    if all_param_vecs:
        big_param = torch.cat(all_param_vecs, dim=0)
        big_act = torch.cat(all_act_vecs, dim=0)

        big_param = big_param / (big_param.norm(p=2) + 1e-12)
        big_act = big_act / (big_act.norm(p=2) + 1e-12)

        global_cos_sim = F.cosine_similarity(big_param, big_act, dim=0).item()
    else:
        global_cos_sim = float("nan")

    return layer_cos_sims, global_cos_sim


########################################
# 4. 使用示例
########################################

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    # 1) 加载 base / defense 模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu"
    )
    defense_model = AutoModelForCausalLM.from_pretrained(
        "/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/different/Llama-3.2-1B-Instruct-tofu-DPO"
    )

    # 2) 概念向量根目录（注意我这里用了 /analysis/results/ 而不是 /analysis_results/）
    concepts_root = "/data/xiangtao/projects/crossdefense/code/analysis/results/01-concepts_vector/Llama-3.2-1B-Instruct-tofu/fast"

    # 3) 计算从第 7 层到第 8 层（只算一层）的对齐度，你也可以把 begin/end 换成 0 / None 算全层
    layer_cos_sims, global_cos_sim = compute_direction_alignment_with_concepts(
        base_model=base_model,
        defense_model=defense_model,
        concepts_root=concepts_root,
        concept_filename="v_safety.pt",  # 如果你有专门的“激活变化方向”文件名，就填那个
        submodule_name="q_proj",
        begin_layer=0,
        end_layer=10,
    )

    print("=== 分层余弦相似度 ===")
    for l in sorted(layer_cos_sims):
        print(f"Layer {l}: cos_sim = {layer_cos_sims[l]:.4f}")

    print("=== 全局拼接后相似度 ===")
    print(global_cos_sim)