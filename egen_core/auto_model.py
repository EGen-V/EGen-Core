import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from egen_core import EGenCoreLlamaMlx

class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )
    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        trust_remote_code = kwargs.get('trust_remote_code', True)
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

        if "Qwen2ForCausalLM" in config.architectures[0]:
            return "egen_core", "EGenCoreQWen2"
        elif "QWen" in config.architectures[0]:
            return "egen_core", "EGenCoreQWen"
        elif "Baichuan" in config.architectures[0]:
            return "egen_core", "EGenCoreBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "egen_core", "EGenCoreChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "egen_core", "EGenCoreInternLM"
        elif "Mistral" in config.architectures[0]:
            return "egen_core", "EGenCoreMistral"
        elif "Mixtral" in config.architectures[0]:
            return "egen_core", "EGenCoreMixtral"
        elif "Llama" in config.architectures[0]:
            return "egen_core", "EGenCoreLlama2"
        else:
            print(f"unknown artichitecture: {config.architectures[0]}, try to use Llama2...")
            return "egen_core", "EGenCoreLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):

        if is_on_mac_os:
            return EGenCoreLlamaMlx(pretrained_model_name_or_path, *inputs, ** kwargs)

        module, cls_name = cls.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, cls_name)
        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)