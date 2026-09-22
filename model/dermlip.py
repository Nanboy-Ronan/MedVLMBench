from open_clip import create_model_from_pretrained, get_tokenizer
from model.openclip_base import OpenCLIPForDiagnosis, OpenCLIPLPForDiagnosis

_DERMLIP_REPO = "hf-hub:redlessone/DermLIP_ViT-B-16"


class _DermLIPMixin:
    def build_model(self):
        return create_model_from_pretrained(_DERMLIP_REPO)

    def tokenize(self, texts):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = get_tokenizer(_DERMLIP_REPO)
        return self._tokenizer(texts)


class DermLIPForDiagnosis(_DermLIPMixin, OpenCLIPForDiagnosis):
    pass


class DermLIPLPForDiagnosis(_DermLIPMixin, OpenCLIPLPForDiagnosis):
    pass
