import json


class CaptionDataset:

    def __init__(self, path, batch_size=30, maximum_sample=50000):
        self.path = path
        self.maximum_sample = maximum_sample
        self.batch_size = batch_size
        self.captions = self.load_captions_dataset()
        self.indices = {key: 0 for key in self.captions}

    def load_captions_dataset(self):
        with open(self.path, 'r') as f:
            captions = json.load(f)
        for k, v in captions.items():
            captions[k] = v[:self.maximum_sample]
        return captions

    def __iter__(self):
        self.indices = {key: 0 for key in self.captions}
        return self

    def __next__(self):
        modalities = []
        captions = []
        num_modalities = sum(1 for key in self.captions
                             if self.indices[key] < len(self.captions[key]))
        if num_modalities == 0:
            raise StopIteration

        per_modality_quota = max(1, self.batch_size // num_modalities)

        for modality, captions_list in self.captions.items():
            start = self.indices[modality]
            end = min(start + per_modality_quota, len(captions_list))
            self.indices[modality] = end

            for caption in captions_list[start:end]:
                modalities.append(modality)
                captions.append(caption)
                if len(modalities) == self.batch_size:
                    break

            if len(modalities) == self.batch_size:
                break

        if not modalities:
            raise StopIteration

        return modalities, captions
