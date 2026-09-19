"""Quilt-LLaVA-v1.5-7B uses the released LLaVA-1.5-7B architecture."""

from model.llava import LLaVA


class QuiltLLaVA(LLaVA):
    def __init__(self, args):
        super().__init__(args)
        self.name = "Quilt-LLaVA-v1.5"
        self.model_type = "medical"
