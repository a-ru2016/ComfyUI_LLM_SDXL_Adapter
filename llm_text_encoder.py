import torch
import logging
try:
    import comfy.model_management
except ImportError:
    pass
from .utils import parse_prompt_weights

logger = logging.getLogger("LLM-SDXL-Adapter")

def find_subsequence(full_seq, sub_seq):
    len_sub = len(sub_seq)
    len_full = len(full_seq)
    if len_sub > len_full:
        return -1
    for i in range(len_full - len_sub + 1):
        if full_seq[i : i + len_sub] == sub_seq:
            return i
    return -1

def interpolate_weights(weights, target_length):
    if not weights or target_length == 0:
        return []
    if len(weights) == target_length:
        return weights
    
    # Use pytorch interpolate
    # weights: list of floats
    w_tensor = torch.tensor(weights, dtype=torch.float32).view(1, 1, -1)
    w_tensor = torch.nn.functional.interpolate(w_tensor, size=target_length, mode='linear', align_corners=False)
    return w_tensor.view(-1).tolist()

class LLMTextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model
    Supports various LLM architectures with chat templates
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLM_MODEL",),
                "tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, 1girl, anime style"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are expert in understanding of user prompts for image generations. Create an image according to the prompt from user."
                }),
                "skip_first": ("INT", {
                    "default": 27,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "max_token_length": ("INT", {
                    "default": 4096,
                    "min": 256,
                    "max": 16384,
                    "step": 64
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "STRING")
    RETURN_NAMES = ("hidden_states", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"
    
    def encode_text(self, model, tokenizer, text, system_prompt="You are expert in understanding of user prompts for image generations. Create an image according to the prompt from user.", skip_first=27, max_token_length=4096):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            # Determine execution device
            try:
                execution_device = comfy.model_management.get_torch_device()
            except:
                execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Check where the model currently is
            model_device = next(model.parameters()).device
            moved_to_gpu = False
            
            # If model is on CPU, move to execution device (GPU) for inference
            if model_device.type == 'cpu' and execution_device.type != 'cpu':
                logger.info(f"Moving model to {execution_device} for inference...")
                model.to(execution_device)
                moved_to_gpu = True
            
            # Prepare chat template
            # Split text by "BREAK" and filter empty strings
            raw_texts = [part.strip() for part in text.split("BREAK") if part.strip()]
            if not raw_texts:
                raw_texts = [""]

            batch_input_ids = []
            max_len = 0

            # Calculate safe chunk length (approximate). 
            # System prompt + overhead takes some space. We reserve roughly 256 tokens for system prompt + wrapper.
            reserved_tokens = 256
            chunk_size = max(64, max_token_length - reserved_tokens)

            for raw_t in raw_texts:
                # Parse weights from text
                clean_text, char_weights = parse_prompt_weights(raw_t)

                # 1. Tokenize the user text solely to check length and get offsets
                # We assume the tokenizer can handle the raw text.
                user_tokens_data = tokenizer(clean_text, add_special_tokens=False, return_offsets_mapping=True)
                user_tokens = user_tokens_data["input_ids"]
                user_offsets = user_tokens_data["offset_mapping"]
                
                # 2. Split user tokens into chunks if necessary
                chunks = []
                chunk_offsets_list = []
                chunk_weights_list = []

                # Pre-calculate token weights from char weights
                full_token_weights = []
                for (start, end) in user_offsets:
                    if start == end:
                        full_token_weights.append(1.0)
                        continue
                    # Handle offsets potentially larger than char_weights length (robustness)
                    end = min(end, len(char_weights))
                    if start < end:
                        segment_weights = char_weights[start:end]
                        if segment_weights:
                            avg_weight = sum(segment_weights) / len(segment_weights)
                            full_token_weights.append(avg_weight)
                        else:
                            full_token_weights.append(1.0)
                    else:
                        full_token_weights.append(1.0)

                if len(user_tokens) > chunk_size:
                    for i in range(0, len(user_tokens), chunk_size):
                        chunks.append(user_tokens[i:i + chunk_size])
                        chunk_offsets_list.append(user_offsets[i:i + chunk_size])
                        chunk_weights_list.append(full_token_weights[i:i + chunk_size])
                else:
                    chunks.append(user_tokens)
                    chunk_offsets_list.append(user_offsets)
                    chunk_weights_list.append(full_token_weights)
                
                # 3. Process each chunk
                chunk_encoded_list = []
                
                for i, chunk_tokens in enumerate(chunks):
                    token_weights = chunk_weights_list[i]
                    
                    # Decode back to text to use with chat template
                    # This is necessary because apply_chat_template expects string, not tokens
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": system_prompt}
                            ]
                        },
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": chunk_text}
                            ]
                        }
                    ]
                    
                    # Apply chat template
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                    )
                    
                    # Get input_ids for this chunk
                    input_ids = encoded.input_ids.to(model.device)
                    
                    # Get embeddings
                    inputs_embeds = model.get_input_embeddings()(input_ids)

                    # Apply weights
                    # Find where chunk_tokens are in input_ids
                    seq_to_find = chunk_tokens
                    input_ids_list = input_ids[0].tolist()
                    
                    start_idx = find_subsequence(input_ids_list, seq_to_find)
                    final_chunk_weights = token_weights
                    target_len = len(seq_to_find)

                    if start_idx == -1:
                        # Try re-tokenized from chunk_text (handling potential decode/encode diffs)
                        retokenized = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
                        seq_to_find = retokenized
                        start_idx = find_subsequence(input_ids_list, seq_to_find)
                        
                        if start_idx != -1:
                            target_len = len(retokenized)
                            final_chunk_weights = interpolate_weights(token_weights, target_len)
                        else:
                            # Fuzzy matching: try ignoring the first token of retokenized sequence
                            # (Common issue: leading space handling in chat templates)
                            if len(retokenized) > 1:
                                sub_seq = retokenized[1:]
                                sub_idx = find_subsequence(input_ids_list, sub_seq)
                                if sub_idx != -1:
                                    # Assuming the missing token is immediately before
                                    start_idx = sub_idx - 1 
                                    target_len = len(retokenized)
                                    final_chunk_weights = interpolate_weights(token_weights, target_len)
                                    
                                    # Correction if start_idx < 0 or logic failure, handled by boundary check below
                            
                            if start_idx == -1:
                                # Log warning but continue
                                logger.warning(f"Could not locate prompt tokens in chat template output. Weights may not be applied correctly for text: '{chunk_text[:20]}...'")

                    if start_idx != -1:
                        # Create a weight tensor for the whole sequence
                        seq_len = input_ids.shape[1]
                        weight_tensor = torch.ones(seq_len, device=model.device, dtype=inputs_embeds.dtype)
                        
                        # Fill in weights
                        # Ensure bounds
                        start_fill = max(0, start_idx)
                        end_fill = min(seq_len, start_idx + target_len)
                        
                        # We need to slice final_chunk_weights if we clipped boundaries
                        w_start = start_fill - start_idx
                        w_end = w_start + (end_fill - start_fill)
                        
                        if w_end > w_start:
                            w_subset = final_chunk_weights[w_start:w_end]
                            weight_tensor[start_fill:end_fill] = torch.tensor(w_subset, device=model.device, dtype=inputs_embeds.dtype)
                                
                        # Multiply embeddings
                        # inputs_embeds: [1, seq_len, hidden_dim]
                        # weight_tensor: [seq_len] -> [1, seq_len, 1]
                        inputs_embeds = inputs_embeds * weight_tensor.view(1, -1, 1)

                    with torch.no_grad():
                        # Pass inputs_embeds instead of input_ids
                        outputs = model(inputs_embeds=inputs_embeds)
                        
                    # Extract hidden states, skipping first tokens
                    # shape: [1, seq_len, hidden_dim]
                    chunk_hidden = outputs['hidden_states'][-1][:, skip_first:, :].to(torch.float)
                    chunk_encoded_list.append(chunk_hidden)

                # Concatenate all chunks for this text entry along sequence dimension
                full_hidden_state = torch.cat(chunk_encoded_list, dim=1)
                batch_input_ids.append(full_hidden_state) # Note: this is now a hidden state tensor, not input_ids
                
            # Concatenate all parts along sequence dimension (dim=1) instead of stacking in batch
            # We assume the user wants one long prompt composed of the BREAK-separated parts
            final_hidden_states = torch.cat(batch_input_ids, dim=1)
            
            # If we moved model to GPU, move it back to CPU to save VRAM
            if moved_to_gpu:
                logger.info("Moving model back to CPU...")
                model.to("cpu")
                torch.cuda.empty_cache()
            
            # Prepare info
            total_tokens = sum(t.shape[1] for t in batch_input_ids)
            info = f"Batch: {len(raw_texts)}\nText: {raw_texts[0][:50]}...\nTotal Encoded Tokens: {total_tokens}\nShape: {final_hidden_states.shape}"
            
            logger.info(f"Encoded text with shape: {final_hidden_states.shape}")
            
            return (final_hidden_states, info)
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMTextEncoder": LLMTextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTextEncoder": "LLM Text Encoder"
}
