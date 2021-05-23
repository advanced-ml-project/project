'''
dataclassCNN.py

File to create a custom data class for pytorch. 

Author: Jesica Maria Ramirez Toscano
'''

from torch.utils.data import Dataset
from nltk import word_tokenize
#import pre_processing as pp


class ProjectDataset(Dataset):

    def __init__(self, data, target_col=None, text_col=None):

        # Target first, then Inputs.
        self.samples = []

        if not target_col and not text_col:
            targets = list(data[0])
            inputs = list(data[1])
            for idx in range(len(targets)):
                text = word_tokenize(inputs[idx])
                self.samples.append([targets[idx], text])
        else:
            for _, row in data.iterrows():
                text = row[text_col]
                #text = pp.clean_text(text, lowercase=False)
                text = word_tokenize(text)
                target = row[target_col]
                self.samples.append([target, text])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
