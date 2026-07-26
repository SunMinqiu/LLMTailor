import json
import os

def get_layer_name(j, num_layers, weight_tie=False):
    """
    根据索引 j 返回对应的层名称。

    索引映射：
    - j == 0: norm
    - 没有 weight_tie:
      - j == num_layers + 1: lm_head
      - j == num_layers + 2: embeddings
      - 其他: attention layer {j-1}
    - 有 weight_tie:
      - j == num_layers + 1: embeddings (lm_head 和 embeddings 共享)
      - 其他: attention layer {j-1}
    """
    if j == 0:
        return "norm"
    if not weight_tie:
        if j == num_layers + 1:
            return "lm_head"
        elif j == num_layers + 2:
            return "embeddings"
        else:
            return f"attention layer {j-1}"
    else:
        if j == num_layers + 1:
            return "embeddings"
        else:
            return f"attention layer {j-1}"

def find_valid_checkpoint(base_path, ckpt_num, max_search=10):
    """
    查找存在的 checkpoint。如果 ckpt_num 对应的 checkpoint 不存在，
    向后搜索直到找到存在的 checkpoint。
    返回实际存在的 checkpoint 编号，如果找不到返回 None。
    """
    for offset in range(max_search):
        candidate = ckpt_num + offset
        candidate_path = f"{base_path}/checkpoint-{candidate}"
        if os.path.exists(candidate_path):
            if offset > 0:
                print(f"checkpoint-{ckpt_num} not found, using checkpoint-{candidate}")
            return candidate
    print(f"WARNING: Cannot find valid checkpoint starting from {ckpt_num}, will use default")
    return None  # 找不到返回 None，让调用者决定用 default

def generate_merge_config_from_log(json_data, base_path, output_file, default_path, end_ckpt, num_layers=24, weight_tie=False):
    # Parse JSON data to get best checkpoint for each layer
    layer_to_ckpt = {}
    
    # Process each step in the JSON data
    for step_data in json_data:
        step = step_data["step"]
        flags = step_data["flag"]
        
        # If step exceeds end_ckpt, skip
        if step > end_ckpt:
            continue
            
        # For each layer that has a flag (indicating it was updated)
        for layer in flags:
            # Update the checkpoint for this layer if we haven't seen it yet
            # or if this step is earlier (better checkpoint)
            if layer not in layer_to_ckpt or step > layer_to_ckpt[layer]:
                layer_to_ckpt[layer] = step
    print("Original layer_to_ckpt:", layer_to_ckpt)

    # 验证并修正所有 checkpoint，确保它们存在
    for layer, ckpt in list(layer_to_ckpt.items()):
        valid_ckpt = find_valid_checkpoint(base_path, ckpt)
        if valid_ckpt is None:
            # 找不到有效 checkpoint，删除这个映射，让它使用 default_path
            del layer_to_ckpt[layer]
        elif valid_ckpt != ckpt:
            layer_to_ckpt[layer] = valid_ckpt

    print("Validated layer_to_ckpt:", layer_to_ckpt)

    # default_path 可能是路径字符串或整数（base_step）
    if isinstance(default_path, int):
        base = default_path
        valid_default = find_valid_checkpoint(base_path, default_path)
        if valid_default is None:
            raise RuntimeError(f"Cannot find valid default checkpoint starting from {default_path}")
        default_path = f"{base_path}/checkpoint-{valid_default}"  # 构造完整路径
    else:
        base = int(default_path.split("-")[-1])
    # Write config file with merged continuous layers (except 0 and last)
    # 根据 weight_tie 确定最大索引
    max_special_idx = num_layers + 1 if weight_tie else num_layers + 2
    with open(output_file, 'w') as f:
        f.write("slices:\n")
        i = 0
        while i <= max_special_idx:
            # For layer 0 and last layer(s), always write individually
            is_special = (i == 0 or i == num_layers + 1 or (not weight_tie and i == num_layers + 2))
            if is_special:
                f.write("  - sources:\n")
                if i in layer_to_ckpt and layer_to_ckpt[i] > base:
                    ckpt = layer_to_ckpt.get(i, 1)
                    f.write(f"      - model: {base_path}/checkpoint-{ckpt}\n")
                    f.write(f"        layer_range: [{i}, {i}]\n")
                else:
                    f.write(f"      - model: {default_path}\n")
                    f.write(f"        layer_range: [{i}, {i}]\n")
                i += 1
                continue

            # For layers 1..num_layers-2, try to merge continuous layers with same ckpt
            start = i
            end = i + 1
            if i in layer_to_ckpt and layer_to_ckpt[i] > base:
                ckpt = layer_to_ckpt.get(i, 1)
                # Only merge if not 0 or last
                while end < num_layers:
                    next_ckpt = layer_to_ckpt.get(end, 1)
                    if next_ckpt == ckpt:
                        end += 1
                    else:
                        break
                # Write merged range
                f.write("  - sources:\n")
                f.write(f"      - model: {base_path}/checkpoint-{ckpt}\n")
                f.write(f"        layer_range: [{start-1}, {end-1}]\n")
            else:
                # Only merge if not 0 or last
                while end < num_layers + 1:
                    if end not in layer_to_ckpt:
                        end += 1
                    else:
                        break
                # Write merged range
                f.write("  - sources:\n")
                f.write(f"      - model: {default_path}\n")
                f.write(f"        layer_range: [{start-1}, {end-1}]\n")
            i = end

        
        f.write("\npost_weights:\n")
        # 索引 0 始终对应 norm
        f.write(f"  - name: model.norm.weight\n")
        if 0 in layer_to_ckpt and layer_to_ckpt[0] > base:
            f.write(f"    model: {base_path}/checkpoint-{layer_to_ckpt[0]}\n")
        else:
            f.write(f"    model: {default_path}\n")

        if not weight_tie:
            # 没有 weight_tie: 索引 num_layers + 1 对应 lm_head
            f.write("  - name: lm_head.weight\n")
            if num_layers + 1 in layer_to_ckpt and layer_to_ckpt[num_layers + 1] > base:
                f.write(f"    model: {base_path}/checkpoint-{layer_to_ckpt[num_layers + 1]}\n")
            else:
                f.write(f"    model: {default_path}\n")

        f.write("\npre_weights:\n")
        f.write(f"  - name: model.embed_tokens.weight\n")
        if not weight_tie:
            # 没有 weight_tie: 索引 num_layers + 2 对应 embeddings
            if num_layers + 2 in layer_to_ckpt and layer_to_ckpt[num_layers + 2] > base:
                f.write(f"    model: {base_path}/checkpoint-{layer_to_ckpt[num_layers + 2]}\n")
            else:
                f.write(f"    model: {default_path}\n")
        else:
            # 有 weight_tie: 索引 num_layers + 1 对应 embeddings
            if num_layers + 1 in layer_to_ckpt and layer_to_ckpt[num_layers + 1] > base:
                f.write(f"    model: {base_path}/checkpoint-{layer_to_ckpt[num_layers + 1]}\n")
            else:
                f.write(f"    model: {default_path}\n")
        f.write("\nmerge_method: passthrough\ndtype: bfloat16")

    print("✅  已写入")