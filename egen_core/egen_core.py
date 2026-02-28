

from .egen_core_base import EGenCoreBaseModel



class EGenCoreLlama2(EGenCoreBaseModel):
    def __init__(self, *args, **kwargs):
        super(EGenCoreLlama2, self).__init__(*args, **kwargs)

