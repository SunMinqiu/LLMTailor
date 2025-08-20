import json
def generate_merge_config_from_log(json_data, base_path, output_file, default_path, end_ckpt, num_layers=24):
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
    print(layer_to_ckpt)
    base = int(default_path.split("-")[-1])
    # Write config file with merged continuous layers (except 0 and last)
    with open(output_file, 'w') as f:
        f.write("slices:\n")
        i = 0
        while i <= num_layers + 2:
            # For layer 0 and last layer, always write individually
            if i == 0 or i == num_layers + 1 or i == num_layers + 2:
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
        f.write(f"  - name: model.norm.weight\n")
        if 0 in layer_to_ckpt and layer_to_ckpt[0] > base:
            f.write(f"    model: {base_path}/checkpoint-{layer_to_ckpt[0]}\n")
        else:
            f.write(f"    model: {default_path}\n")
        f.write("  - name: lm_head.weight\n")
        if num_layers + 1 in layer_to_ckpt and layer_to_ckpt[num_layers + 1] > base:
            f.write(f"    model: {base_path}/checkpoint-{layer_to_ckpt[num_layers + 1]}\n")
        else:
            f.write(f"    model: {default_path}\n")
    
        f.write("\npre_weights:\n")
        f.write(f"  - name: model.embed_tokens.weight\n")
        if num_layers + 2 in layer_to_ckpt and layer_to_ckpt[num_layers + 2] > base:
            f.write(f"    model: {base_path}/checkpoint-{layer_to_ckpt[num_layers + 2]}\n")
        else:
            f.write(f"    model: {default_path}\n")
        f.write("\nmerge_method: passthrough\ndtype: bfloat16")

    print("✅  已写入")