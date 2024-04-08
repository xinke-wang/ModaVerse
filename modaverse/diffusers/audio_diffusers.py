import scipy
import torch
from diffusers import AudioLDMPipeline

from modaverse.diffusers import DIFFUSER_REGISTRY

from .base import BaseDiffuser


@DIFFUSER_REGISTRY.register('audio_ldm')
class AudioLDM(BaseDiffuser):

    output_modality = 'audio'

    def __init__(self, config) -> None:
        super().__init__(config)

    def load_model(self) -> AudioLDMPipeline:
        model = AudioLDMPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
            if self.half_precision else torch.float32)
        model = model.to('cuda')

        return model

    def _generate(self,
                  prompt: str,
                  num_inference_steps: int = 10,
                  audio_length_in_s: float = 5.0) -> str:
        return self.model(prompt,
                          num_inference_steps=num_inference_steps,
                          audio_length_in_s=audio_length_in_s).audios[0]

    def dump(self, generated, dst_path: str, rate: int = 16000) -> None:
        scipy.io.wavfile.write(dst_path, rate, generated)
