import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import logging
from .utils import get_llm_checkpoints, get_llm_checkpoint_path

logger = logging.getLogger("LLM-SDXL-Adapter")


class LLMModelLoader:
    """
    ComfyUI node that loads Language Model and tokenizer
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_llm_checkpoints(), {
                    "default": get_llm_checkpoints()[0] if get_llm_checkpoints() else None
                }),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {
                    "default": "auto"
                }),
                "dtype": (["auto", "fp16", "bf16", "fp32"], {
                    "default": "auto"
                }),
                "quantization": (["none", "8bit", "4bit"], {
                    "default": "none"
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_MODEL", "LLM_TOKENIZER", "STRING")
    RETURN_NAMES = ("model", "tokenizer", "info")
    FUNCTION = "load_model"
    CATEGORY = "llm_sdxl"
    
    def load_model(self, model_name, device="auto", dtype="auto", quantization="none", force_reload=False):
        """Load Language Model and tokenizer"""
        if device == "auto":
            device = self.device
                
        try:
            model_path = get_llm_checkpoint_path(model_name)

            # Check if we need to reload
            if force_reload or self.model is None or self.current_model_path != model_path:
                # Clear previous model
                if self.model is not None:
                    del self.model
                    del self.tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading Language Model from {model_path}")
                
                # Prepare quantization config
                quantization_config = None
                if quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                elif quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                
                # Prepare torch dtype
                torch_dtype = "auto"
                if dtype == "fp16":
                    torch_dtype = torch.float16
                elif dtype == "bf16":
                    torch_dtype = torch.bfloat16
                elif dtype == "fp32":
                    torch_dtype = torch.float32

                # If quantized, device_map must be handled by accelerate usually, but we try to respect 'device' if not 'auto'
                # For 4bit/8bit, device_map="auto" or specific device is often needed.
                load_device_map = device
                if quantization != "none":
                     # When quantized, 'auto' is usually best to let accelerate handle dispatch
                     if device == "cpu":
                         logger.warning("Quantization on CPU is not supported by bitsandbytes. Ignoring quantization or expects GPU.")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                    device_map=load_device_map,
                    output_hidden_states=True,
                    trust_remote_code=True
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                self.current_model_path = model_path
                logger.info(f"Language Model loaded successfully. Device: {self.model.device}, Dtype: {self.model.dtype}, Quantization: {quantization}")
            
            info = f"Model: {model_path}\nDevice: {device}\nDtype: {dtype}\nQuantization: {quantization}\nLoaded: {self.model is not None}"
            
            return (self.model, self.tokenizer, info)
            
        except Exception as e:
            logger.error(f"Failed to load Language Model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")



# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMModelLoader": LLMModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMModelLoader": "LLM Model Loader",
} 