'''
dataclassCNN.py
File to create a Custom Data Class and Collate Function for PyTorch.
This file is for the CNN model.
'''
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


class ProjectDataset(Dataset):

    def __init__(self, data, target_col=None, text_col=None):

        # Target first, then Inputs.
        self.samples = []
        tokenizer = get_tokenizer('basic_english')

        if not target_col and not text_col:
            targets = list(data[0])
            inputs = list(data[1])
            for idx in range(len(targets)):
                text = tokenizer(inputs[idx])
                self.samples.append([targets[idx], text])
        else:
            for _, row in data.iterrows():
                text = row[text_col]
                text = tokenizer(text)
                target = row[target_col]
                self.samples.append([target, text])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
