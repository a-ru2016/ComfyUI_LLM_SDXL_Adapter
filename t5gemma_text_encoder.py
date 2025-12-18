import torch
import logging
from .utils import parse_prompt_weights

logger = logging.getLogger("LLM-SDXL-Adapter")


class T5GEMMATextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model
    Supports various LLM architectures with chat templates
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {"multiline": True, "default": "masterpiece, best quality, 1girl, anime style"}),
                "max_length": ("INT", {"default": 1024, "min": 8, "max": 8192}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
                "dtype": (["float32", "bfloat16"], {"default": "bfloat16"}),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "LLM_ATTENTION_MASK", "STRING")
    RETURN_NAMES = ("hidden_states", "attention_mask", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"
    
    def encode_text(self, llm_model, llm_tokenizer, text, max_length, device, dtype):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            target_device = torch.device(device)
            target_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

            # Check where the model currently is
            original_device = next(llm_model.parameters()).device
            moved_model = False

            # If model is not on the target execution device, move it
            if original_device.type != target_device.type:
                logger.info(f"Moving model from {original_device} to {target_device} for inference...")
                llm_model.to(target_device)
                moved_model = True

            # Tokenize
            # Split text by "BREAK" and filter empty strings
            raw_texts = [part.strip() for part in text.split("BREAK") if part.strip()]
            if not raw_texts:
                raw_texts = [""]
            
            clean_texts = []
            char_weights_list = []

            for rt in raw_texts:
                c, w = parse_prompt_weights(rt)
                clean_texts.append(c + "<eos>")
                # Extend weights for <eos> suffix (approx 5 chars length to be safe)
                w.extend([1.0] * 5)
                char_weights_list.append(w)

            inputs = llm_tokenizer(
                clean_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True
            )
            input_ids = inputs.input_ids.to(target_device)
            attention_mask = inputs.attention_mask.to(target_device)
            offset_mapping = inputs.offset_mapping 
            logger.info(f"Tokenized prompt length: {input_ids.shape[1]} (batch size: {input_ids.shape[0]})")

            # Generate hidden states
            with torch.no_grad():
                # Get embeddings
                try:
                    embed_layer = llm_model.get_input_embeddings()
                except AttributeError:
                     # Fallback for models where get_input_embeddings is not available or named differently
                     # But usually HF models have this.
                     raise Exception("Model does not support get_input_embeddings()")
                
                inputs_embeds = embed_layer(input_ids)

                # Calculate token weights
                batch_size, seq_len = input_ids.shape
                # Match the dtype and device of the embeddings to avoid mismatches
                weight_tensor = torch.ones((batch_size, seq_len, 1), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                
                for i in range(batch_size):
                    c_weights = char_weights_list[i]
                    offsets = offset_mapping[i]
                    
                    for j in range(seq_len):
                        start, end = offsets[j]
                        if start == end: # Special token or padding
                            continue
                        # Safely handle bounds
                        end = min(end, len(c_weights))
                        
                        if start < end:
                            segment = c_weights[start:end]
                            if segment:
                                avg = sum(segment) / len(segment)
                                weight_tensor[i, j, 0] = avg

                inputs_embeds = inputs_embeds * weight_tensor
                
                # Ensure model is in correct dtype if possible
                outputs = llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                # Extract hidden states
                hidden_states = outputs.last_hidden_state.to(torch.float32) # SDXL expects float32 usually for mixing
            
            # Combine parts split by BREAK into one long sequence
            # We need to remove padding from each part and concatenate them along dim 1
            valid_hidden_states_list = []
            valid_masks_list = []
            
            for i in range(hidden_states.shape[0]):
                # Get valid length from attention mask
                valid_len = int(attention_mask[i].sum().item())
                
                # Slice hidden states and mask
                # hidden_states: [Batch, Seq, Dim] -> [1, Valid_Seq, Dim]
                valid_hidden_states_list.append(hidden_states[i, :valid_len, :].unsqueeze(0))
                # attention_mask: [Batch, Seq] -> [1, Valid_Seq]
                valid_masks_list.append(attention_mask[i, :valid_len].unsqueeze(0))

            final_hidden_states = torch.cat(valid_hidden_states_list, dim=1)
            final_attention_mask = torch.cat(valid_masks_list, dim=1)

            # If we moved model, move it back to original device (likely CPU) to save VRAM
            if moved_model:
                logger.info(f"Moving model back to {original_device}...")
                llm_model.to(original_device)
                if target_device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate actual token count (sum of attention mask)
            token_count = int(final_attention_mask.sum().item())

            # Prepare info
            info = f"Batch: {len(clean_texts)}\nText: {clean_texts[0][:50]}...\nTokens: {token_count}\nEncoded: {final_hidden_states.shape[1]}\nShape: {final_hidden_states.shape}"
            
            logger.info(f"Encoded text with shape: {final_hidden_states.shape}, Tokens: {token_count}")
            
            return (final_hidden_states, final_attention_mask, info)
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMATextEncoder": T5GEMMATextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMATextEncoder": "T5Gemma Text Encoder"
} 