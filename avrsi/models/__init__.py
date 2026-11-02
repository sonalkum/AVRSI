from .base import ACTOR, CRITIC, REFERENCE, Backend, Generation, ModelInput, Sampling, build_model_input


def load_backend(config: dict, device: str = "cuda") -> Backend:
    """Real wrappers, UNTESTED on real weights."""
    from .avf import AVFBackend
    from .qwen import Qwen2_5OmniBackend, Qwen3OmniBackend

    classes = {"qwen3_omni": Qwen3OmniBackend, "qwen2_5_omni": Qwen2_5OmniBackend, "avf": AVFBackend}
    return classes[config["backend"]](config, device)


__all__ = ["ACTOR", "CRITIC", "REFERENCE", "Backend", "Generation", "ModelInput", "Sampling", "build_model_input", "load_backend"]
