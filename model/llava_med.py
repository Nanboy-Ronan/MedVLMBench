from model.llava import LLaVA


class LLaVAMed(LLaVA):
    def __init__(self, args):
        super().__init__(args)

        self.name = "LLaVA-Med"
