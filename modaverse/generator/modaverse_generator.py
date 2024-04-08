from typing import Any

from modaverse.diffusers import DIFFUSER_REGISTRY


class ModaVerseGenerator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.load_diffusers()

    def load_diffusers(self):
        activated_modality = []
        image_diffuser = getattr(self.cfg, 'image_diffuser', None)
        if image_diffuser is not None:
            self.image_diffuser = DIFFUSER_REGISTRY.get(image_diffuser.type)(
                image_diffuser.cfgs)
            activated_modality.append('[IMAGE]')
        video_diffuser = getattr(self.cfg, 'video_diffuser', None)
        if video_diffuser is not None:
            self.video_diffuser = DIFFUSER_REGISTRY.get(video_diffuser.type)(
                video_diffuser.cfgs)
            activated_modality.append('[VIDEO]')
        audio_diffuser = getattr(self.cfg, 'audio_diffuser', None)
        if audio_diffuser is not None:
            self.audio_diffuser = DIFFUSER_REGISTRY.get(audio_diffuser.type)(
                audio_diffuser.cfgs)
            activated_modality.append('[AUDIO]')

        self.active_modality = activated_modality

    def generate(self, modality, text):
        if modality not in self.active_modality:
            raise ValueError(f'Modality {modality} is not activated!')

        if modality == '[IMAGE]':
            return self.image_diffuser.generate(text)
        elif modality == '[VIDEO]':
            return self.video_diffuser.generate(text)
        elif modality == '[AUDIO]':
            return self.audio_diffuser.generate(text)

        return

    def __call__(self, modality: str, prompt: str) -> Any:
        return self.generate(modality, prompt)
