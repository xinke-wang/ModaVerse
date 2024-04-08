import torch
from diffusers import StableDiffusionPipeline

from modaverse.diffusers import DIFFUSER_REGISTRY

from .base import BaseDiffuser


@DIFFUSER_REGISTRY.register('stable_diffusion')
class StableDiffusion(BaseDiffuser):

    output_modality = 'image'

    def __init__(self, config) -> None:
        super().__init__(config)

    def load_model(self) -> StableDiffusionPipeline:
        model = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
            if self.half_precision else torch.float32)
        model = model.to('cuda')

        return model

    def _generate(self, prompt: str) -> str:
        return self.model(prompt).images[0]

    def dump(self, generated, dst_path) -> None:
        generated.save(dst_path)
