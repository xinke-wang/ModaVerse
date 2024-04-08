import os.path as osp
from typing import List, Optional

import torch
from pjtools.configurator import AutoConfigurator

from modaverse.generator import ModaVerseGenerator
from modaverse.model import ModaVerse
from modaverse.parser import ModaVerseParser


class ModaVerseAPI:

    def __init__(self,
                 model_path: str = '.checkpoints/ModaVerse-7b',
                 cfg_path: str = 'configs/base.py',
                 half_precision: bool = True,
                 meta_response_only: bool = False) -> None:
        configs = AutoConfigurator.fromfile(cfg_path)
        self.model = ModaVerse(configs)
        self.model.load_state_dict(torch.load(osp.join(model_path,
                                                       'pytorch_model.pt'),
                                              map_location='cuda'),
                                   strict=False)
        self.model.eval()
        if half_precision:
            self.model = self.model.half()
        self.meta_response_only = meta_response_only
        self.model = self.model.cuda()
        self.parser = ModaVerseParser(configs.model_configs.modaverse,
                                      self.model.tokenizer)
        if not self.meta_response_only:
            self.generator = ModaVerseGenerator(
                configs.model_configs.modaverse.generator)

    def __call__(self,
                 instruction: str,
                 input_media: Optional[List[str]] = []) -> List[dict]:
        with torch.no_grad():
            meta_response = self.model.generate(instruction, input_media)
        modalities, prompts = self.parser.parse(meta_response)
        final_response = []
        if not self.meta_response_only:
            for modality, prompt in zip(modalities, prompts):
                if modality == '[TEXT]':
                    final_response.append({'type': 'text', 'content': prompt})
                else:
                    final_response.append(self.generator(modality, prompt))

        return meta_response, final_response
