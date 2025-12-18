import os
import logging
import folder_paths

logger = logging.getLogger("LLM-SDXL-Adapter")

def get_llm_dict():
    """
    Get the dictionary of LLM checkpoints.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """
    llm_dict = {}
    if "llm" in folder_paths.folder_names_and_paths:
        llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
    elif os.path.exists(os.path.join(folder_paths.models_dir, "llm")):
        llm_paths = [os.path.join(folder_paths.models_dir, "llm")]
    else:
        llm_paths = [os.path.join(folder_paths.models_dir, "LLM")]

    for llm_path in llm_paths:
        if os.path.exists(llm_path):
            for item in os.listdir(llm_path):
                item_path = os.path.join(llm_path, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid model directory (contains config.json or similar)
                    if any(f in os.listdir(item_path) for f in ['config.json', 'model.safetensors', 'pytorch_model.bin']):
                        llm_dict[item] = item_path
                elif item.endswith(('.safetensors', '.bin', '.pt')):
                    llm_dict[item] = item_path

    return llm_dict

def get_llm_gguf_dict():
    """
    Get the dictionary of GGUF files.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """
    llm_gguf_dict = {}
    if "llm" in folder_paths.folder_names_and_paths:
        llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
    elif os.path.exists(os.path.join(folder_paths.models_dir, "llm")):
        llm_paths = [os.path.join(folder_paths.models_dir, "llm")]
    else:
        llm_paths = [os.path.join(folder_paths.models_dir, "LLM")]

    for llm_path in llm_paths:
        if os.path.exists(llm_path):
            for item in os.listdir(llm_path):
                item_path = os.path.join(llm_path, item)
                if os.path.isfile(item_path):
                    if item_path.lower().endswith('.gguf'):
                        llm_gguf_dict[item] = llm_path

    return llm_gguf_dict
    
def get_adapters_dict():
    """
    Get the dictionary of LLM adapters.
    Keys are the names of the LLM adapters, values are the paths to the LLM adapters.
    """
    adapters_dict = {}
    if "llm_adapters" in folder_paths.folder_names_and_paths:
        adapters_paths, _ = folder_paths.folder_names_and_paths["llm_adapters"]
    else:
        adapters_paths = [os.path.join(folder_paths.models_dir, "llm_adapters")]

    for adapters_path in adapters_paths:
        if os.path.exists(adapters_path):
            for item in os.listdir(adapters_path):
                if item.endswith('.safetensors'):
                    adapters_dict[item] = os.path.join(adapters_path, item)

    return adapters_dict

def get_llm_checkpoints():
    """
    Get the list of available LLM checkpoints.
    """
    return list(get_llm_dict().keys())

def get_llm_ggufs():
    """
    Get the list of available LLM checkpoints packed in GGUF.
    """
    return list(get_llm_gguf_dict().keys())

def get_llm_adapters():
    """
    Get the list of available LLM adapters.
    """
    return list(get_adapters_dict().keys())

def get_llm_checkpoint_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")

def get_llm_gguf_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_gguf_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")

def get_llm_adapter_path(adapter_name):
    """
    Get the path to an LLM adapter.
    """
    adapters_dict = get_adapters_dict()

    if adapter_name in adapters_dict:
        return adapters_dict[adapter_name]
    else:
        raise ValueError(f"Adapter {adapter_name} not found")

def parse_prompt_weights(text):
    """
    Parses text with ComfyUI-style weights: (text:1.2), (text), [text].
    Returns (cleaned_text, list_of_weights_per_char).
    """
    chars = list(text)
    weights = [1.0] * len(text)
    
    # Stack stores: (start_index, type)
    # Types: 'paren', 'bracket'
    stack = []
    to_remove = [False] * len(text)
    
    i = 0
    while i < len(chars):
        c = chars[i]
        if c == '(':
            stack.append((i, 'paren'))
        elif c == '[':
            stack.append((i, 'bracket'))
        elif c == ')' and stack:
            # Find matching opening
            # Since we want to match the most recent opener, we look at the top of stack
            if stack[-1][1] == 'paren':
                start_i, _ = stack.pop()
                
                # Check for :weight inside
                # Search for ':' between start_i and i, backwards
                colon_idx = -1
                for k in range(i - 1, start_i, -1):
                    if chars[k] == ':' and not to_remove[k]:
                         # Found a colon that hasn't been removed (so it's at this level)
                         colon_idx = k
                         break
                
                weight = 1.1
                content_end = i
                
                if colon_idx != -1:
                    try:
                        # Try to parse number after colon
                        # We need to construct the string carefully ignoring removed chars?
                        # Actually, in standard syntax, the number is immediate.
                        # (tag:1.2) -> tag is start_i+1 to colon_idx
                        # 1.2 is colon_idx+1 to i
                        # We should verify the number part only contains the number
                        w_str = "".join([chars[k] for k in range(colon_idx+1, i) if not to_remove[k]])
                        weight = float(w_str)
                        
                        # Mark colon and number as removed
                        for k in range(colon_idx, i):
                            to_remove[k] = True
                        content_end = colon_idx
                    except ValueError:
                        # Not a valid number, treat as normal parens
                        pass
                
                # Apply weight to content
                for k in range(start_i + 1, content_end):
                    if not to_remove[k]:
                        weights[k] *= weight
                
                # Mark parens as removed
                to_remove[start_i] = True
                to_remove[i] = True
            else:
                # Mismatched or interleaving, e.g. ([) ]
                # If we hit ) but top is [, we ignore it or pop until we find (?
                # ComfyUI usually is strict or just ignores.
                # Let's just ignore this closing char if it doesn't match top
                pass
                
        elif c == ']' and stack:
            if stack[-1][1] == 'bracket':
                start_i, _ = stack.pop()
                weight = 0.9
                for k in range(start_i + 1, i):
                    if not to_remove[k]:
                        weights[k] *= weight
                to_remove[start_i] = True
                to_remove[i] = True
            else:
                pass
        
        i += 1
        
    # Reconstruct
    res_text = ""
    res_weights = []
    for i in range(len(chars)):
        if not to_remove[i]:
            res_text += chars[i]
            res_weights.append(weights[i])
            
    return res_text, res_weights
