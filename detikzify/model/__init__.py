from transformers import AutoConfig, AutoModel
from types import SimpleNamespace
from datasets import DownloadManager
from transformers.utils.hub import is_remote_url
from transformers import AutoTokenizer, PretrainedConfig

from .modeling_detikzify import DetikzifyConfig, DetikzifyForCausalLM

def register():
    try:
        AutoConfig.register("detikzify", DetikzifyConfig)
        AutoModel.register(DetikzifyConfig, DetikzifyForCausalLM)
    except ValueError:
        pass # already registered

def load_deepseek(size="1.3b", **kwargs):
    return load(
        base_model=f"deepseek-ai/deepseek-coder-{size}-base",
        **kwargs
    )

def load_codellama(size="7b", **kwargs):
    return load(
        base_model=f"codellama/CodeLlama-{size}-hf",
        **kwargs
    )

def load(base_model, vision_tower="vit_so400m_patch14_siglip_384.webli", pretrain_mm_mlp_adapter=None, **kwargs):
    tokenizer_name_or_path = PretrainedConfig.from_pretrained(base_model).name_or_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,
        model_max_length=2048,
        add_bos_token=False,
        add_eos_token=True,
        pad_token="<pad>",
        padding_side="right", # Note: only for training, need to change to "left" for batched inference
        legacy=False
    )
    model = DetikzifyForCausalLM.from_pretrained(base_model,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs
    )

    model.config.model_type = DetikzifyConfig.model_type # type: ignore
    if len(tokenizer) > model.config.vocab_size: # type: ignore
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8) # type: ignore

    if pretrain_mm_mlp_adapter and is_remote_url(pretrain_mm_mlp_adapter):
        pretrain_mm_mlp_adapter = DownloadManager().download(pretrain_mm_mlp_adapter)

    processor = model.get_model().initialize_vision_modules( # type: ignore
        vision_tower=vision_tower,
        patch_token_id=tokenizer.bos_token_id,
        pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
    )

    return model, SimpleNamespace(text=tokenizer, image=processor)
