'''
dataobject.py

File to create a custom data class for pytorch. 

Author: Jesica Maria Ramirez Toscano
'''

from torch.utils.data import Dataset
from nltk import word_tokenize
import pre_processing as pp


class ProjectDataset(Dataset):

    def __init__(self, data, target_col, text_col):

        data_lists = []

        for index, row in data.iterrows():
            text = pp.clean_text(row[text_col], lowercase=False)
            text = word_tokenize(text)
            target = row[target_col]
            data_lists.append([target, text])

        self.samples = data_lists

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
