
from transformers import GenerationConfig

from .egen_core_base import EGenCoreBaseModel



class EGenCoreInternLM(EGenCoreBaseModel):


    def __init__(self, *args, **kwargs):


        super(EGenCoreInternLM, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


