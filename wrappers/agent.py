from model.chat import ChatMetaModel


class AgentMetaWrapper(ChatMetaModel):
    def __init__(self, backbone, args):
        super().__init__(args)

        self.backbone = backbone
        self.name = f"Agent({getattr(backbone, 'name', args.model)})"
        self.model_type = getattr(backbone, "model_type", "")
        self.last_trace = {}

        self._sync_backbone_state()

    def _sync_backbone_state(self):
        self.model = getattr(self.backbone, "model", None)
        self.tokenizer = getattr(self.backbone, "tokenizer", None)
        self.processor = getattr(self.backbone, "processor", None)

        self.image_processor = getattr(self.backbone, "image_processor", None)
        self.image_processor_callable = getattr(self.backbone, "image_processor_callable", None)

    def load_from_pretrained(self, model_path, **kwargs):
        self.backbone.load_from_pretrained(model_path, **kwargs)
        self._sync_backbone_state()

    def load_for_training(self, model_path):
        self.backbone.load_for_training(model_path)
        self._sync_backbone_state()

    def save(self, output_folder, trainer=None):
        if trainer is None:
            return self.backbone.save(output_folder)
        return self.backbone.save(output_folder, trainer=trainer)

    def set_inference_context(self, context=None):
        super().set_inference_context(context)
        if hasattr(self.backbone, "set_inference_context"):
            self.backbone.set_inference_context(context)

    def set_device(self, device):
        super().set_device(device)
        if hasattr(self.backbone, "set_device"):
            self.backbone.set_device(device)
        else:
            self.backbone.device = device

    def get_last_trace(self):
        return self.last_trace

    def reset(self):
        self.last_trace = {}

    def _query_backbone(self, image, prompt, image_size=None, temperature=None):
        """Run inference of backbone VLM to get response"""
        return self.backbone.infer_vision_language(image, prompt, image_size, temperature=temperature).strip()
