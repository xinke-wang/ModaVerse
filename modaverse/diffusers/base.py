import os.path as osp
from typing import Any
from uuid import uuid4

import torch
from pjtools.registry import Registry

from modaverse.utils import mkdir_if_not_exist

DIFFUSER_REGISTRY = Registry('diffuser')


class BaseDiffuser:

    output_modality = None
    format_map = {'image': 'png', 'video': 'webm', 'audio': 'wav'}

    def __init__(self, config):
        self.config = config.to_dict()
        self.half_precision = getattr(self.config, 'half_precision', True)
        self.init_args = getattr(self.config, 'init_args', {})
        self.output_path = getattr(self.config, 'output_path', 'outputs')
        mkdir_if_not_exist(self.output_path)
        self.model_name = config.model
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str) -> Any:
        dst_path = osp.join(
            self.output_path,
            f'{uuid4()}.{self.format_map[self.output_modality]}')
        with torch.no_grad():
            generated = self._generate(prompt)
        self.dump(generated, dst_path)
        return {'type': self.output_modality, 'content': dst_path}

    def _generate(self, prompt: str) -> Any:
        raise NotImplementedError

    def dump(self, generated: Any, dst_path: str):
        raise NotImplementedError

    def __call__(self, prompt: str) -> Any:
        return self.generate(prompt)
