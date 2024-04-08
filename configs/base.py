from typing import List

import torch
from peft import LoraConfig, TaskType
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


prompt_configs = dict(path='assets/prompts/prompt_template.txt',
                      media_placeholder='{media}',
                      instruction_placeholder='{instruction}')

model_configs = dict(
    name='ModaVerse-7b',
    imagebind=dict(hidden_size=1024),
    foundation_llm=dict(type='vicuna-7b', checkpoint='.checkpoints/7b_v0'),
    modaverse=dict(
        max_length=512,
        modality_begin_token='<Media>',
        modality_end_token='</Media>',
        modality_flags=['[TEXT]', '[IMAGE]', '[AUDIO]', '[VIDEO]'],
        target_padding=-100,
        top_p=0.01,
        temperature=1,
        max_new_tokens=246,
        do_sample=True,
        use_cache=True,
        stopping_token=835,
        stopping_criteria=StoppingCriteriaList(
            [StoppingCriteriaSub(stops=[[835]], encounters=1)], ),
        generator=dict(
            image_diffuser=dict(
                type='stable_diffusion',
                # preload=False,
                cfgs=dict(model='runwayml/stable-diffusion-v1-5')),
            video_diffuser=dict(
                type='damo_vilab',
                # preload=False,
                cfgs=dict(model='damo-vilab/text-to-video-ms-1.7b')),
            audio_diffuser=dict(type='audio_ldm',
                                cfgs=dict(model='cvssp/audioldm-l-full')),
        ),
    ))

training_configs = dict(
    lora_config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']),
    deepspeed_cfg=dict(path='configs/dscfg.json', backend='nccl'),
    saving_root='./experiments',
    epochs=1,
    warmup_rate=0.1,
    force_training_layers=['embed_tokens.weight', 'lm_head.weight'],
    report_backend=dict(type='wandb', iterval=10),
    print_prediction=dict(turn_on=True, interval=1000),
    checkpointer=dict(type='iteration', interval=5000))

dataset_configs = dict(train=dict(instruction_path='dataset/instructions.json',
                                  media_root='dataset/'))
