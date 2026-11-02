"""A tiny random config of the real Qwen3-Omni thinker class: real token ids, tiny dimensions."""
import torch
from transformers import Qwen3OmniMoeThinkerForConditionalGeneration as Thinker
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig

LORA_REGEX = r"model\.layers\.\d+\.self_attn\.[qkvo]_proj"
LAYERS = 2
BASE = {"backend": "qwen3_omni", "model_id": "tiny", "audio_in_video_flag": "use_audio_in_video", "dtype": "bf16", "missing_modality_fill": None,
        "lora_target_modules": LORA_REGEX, "num_frames": None, "audio_window_s": None, "image_size": None}


def tiny_config(vocab_size=152064, attn="sdpa"):
    text = dict(vocab_size=vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=LAYERS, num_attention_heads=2, num_key_value_heads=2,
                moe_intermediate_size=8, num_experts=4, num_experts_per_tok=2, head_dim=8, max_position_embeddings=4096,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1], "interleaved": True})
    audio = dict(num_mel_bins=128, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=16, d_model=16, output_dim=16, n_window=50,
                 n_window_infer=800, downsample_hidden_size=8)
    vision = dict(depth=2, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=16, deepstack_visual_indexes=[0, 1], patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2)
    config = Qwen3OmniMoeThinkerConfig(audio_config=audio, vision_config=vision, text_config=text, audio_token_id=151675, image_token_id=151655,
                                       video_token_id=151656, audio_start_token_id=151669, position_id_per_seconds=13)
    config.audio_end_token_id, config.vision_start_token_id, config.vision_end_token_id = 151670, 151652, 151653
    config._attn_implementation = attn
    return config


def tiny_thinker(seed=0, dtype=torch.bfloat16, **kwargs):
    torch.manual_seed(seed)
    return Thinker(tiny_config(**kwargs)).to(dtype).eval()
