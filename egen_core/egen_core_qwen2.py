
from transformers import GenerationConfig


from .egen_core_base import EGenCoreBaseModel



class EGenCoreQWen2(EGenCoreBaseModel):


    def __init__(self, *args, **kwargs):


        super(EGenCoreQWen2, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False


