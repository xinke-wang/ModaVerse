import re


class ModaVerseParser:

    def __init__(self, cfg, tokenizer) -> None:
        self.modality_flags = cfg.modality_flags
        self.terminate_flag = tokenizer.decode(cfg.stopping_token)

    def parse(self,
              meta_response: str,
              default_modality: str = '[TEXT]',
              regex=r'(?s)(?:\[[^\]]*\]\s*)*\bPrompt:\s*(.+?)(?:\s*###|$)'):
        flag_counts = {
            flag: meta_response.count(flag)
            for flag in self.modality_flags
        }
        modality = [
            k for k, v in sorted(
                flag_counts.items(), key=lambda item: item[1], reverse=True)
            if v > 0
        ]
        if len(modality) == 0:
            modality = [default_modality]
        prompt = re.search(regex, meta_response).group(1).strip()
        prompt = len(modality) * [prompt]

        return modality, prompt
