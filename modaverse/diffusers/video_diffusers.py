import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from modaverse.diffusers import DIFFUSER_REGISTRY
from modaverse.utils import export_to_video

from .base import BaseDiffuser


@DIFFUSER_REGISTRY.register('damo_vilab')
class DamoVilabTextToVideo(BaseDiffuser):

    output_modality = 'video'

    def __init__(self, config) -> None:
        super().__init__(config)

    def load_model(self) -> DiffusionPipeline:
        model = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
            if self.half_precision else torch.float32,
            variant='fp16')
        model.scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config)
        model.enable_model_cpu_offload()
        model = model.to('cuda')

        return model

    def _generate(self, prompt: str, num_inference_steps: int = 25) -> str:
        return self.model(prompt,
                          num_inference_steps=num_inference_steps).frames[0]

    def dump(self, generated, dst_path) -> None:
        export_to_video(generated, dst_path)
