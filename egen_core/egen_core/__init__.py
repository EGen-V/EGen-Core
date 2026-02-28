from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .egen_core_llama_mlx import EGenCoreLlamaMlx
    from .auto_model import AutoModel
else:
    from .egen_core import EGenCoreLlama2
    from .egen_core_chatglm import EGenCoreChatGLM
    from .egen_core_qwen import EGenCoreQWen
    from .egen_core_qwen2 import EGenCoreQWen2
    from .egen_core_baichuan import EGenCoreBaichuan
    from .egen_core_internlm import EGenCoreInternLM
    from .egen_core_mistral import EGenCoreMistral
    from .egen_core_mixtral import EGenCoreMixtral
    from .egen_core_base import EGenCoreBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException

