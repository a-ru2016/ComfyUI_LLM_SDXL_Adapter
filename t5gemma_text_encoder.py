import torch
import logging

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
            inputs = llm_tokenizer(
                text + "<eos>",
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
            input_ids = inputs.input_ids.to(target_device)
            attention_mask = inputs.attention_mask.to(target_device)
            logger.info(f"Tokenized prompt length: {input_ids.shape[1]} (batch size: {input_ids.shape[0]})")

            # Generate hidden states
            with torch.no_grad():
                # Ensure model is in correct dtype if possible (though .to(dtype) on model might be heavy, usually just casting output is enough or model is already loaded in correct dtype)
                # We trust the loader set the correct dtype, but here we cast output.
                outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                # Extract hidden states
                hidden_states = outputs.last_hidden_state.to(torch.float32) # SDXL expects float32 usually for mixing
                
            # If we moved model, move it back to original device (likely CPU) to save VRAM
            if moved_model:
                logger.info(f"Moving model back to {original_device}...")
                llm_model.to(original_device)
                if target_device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate actual token count (sum of attention mask)
            token_count = int(inputs.attention_mask.sum().item())

            # Prepare info
            info = f"Text: {text[:50]}...\nTokens: {token_count}\nEncoded: {hidden_states.shape[1]}\nShape: {hidden_states.shape}"
            
            logger.info(f"Encoded text with shape: {hidden_states.shape}, Tokens: {token_count}")
            
            return (hidden_states, attention_mask, info)
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