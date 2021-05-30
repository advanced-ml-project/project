'''
dataobject.py

File to create a custom data class for pytorch. 

Author: Jesica Maria Ramirez Toscano
'''

from torch.utils.data import Dataset
from nltk import word_tokenize
import pre_processing as pp
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.vocab import GloVe

class ProjectDataset(Dataset):

    def __init__(self, data, target_col, text_col, split=0.2, random_seed=42):

        data_lists = []

        for index, row in data.iterrows():
            text = pp.clean_text(row[text_col], lowercase=False)
            text = word_tokenize(text)
            target = row[target_col]
            data_lists.append([target, text])

        self.samples = data_lists
        
        # creating train/test/validation indices
        np.random.seed(random_seed)
        dataset_size = len(data)
        indices = list(range(dataset_size))
        split_num = int(np.floor(split * dataset_size))
        
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split_num:], indices[:split_num]
        
        # setting half of validation aside for testing
        val_split = int(np.floor(0.5 * len(val_indices)))
        np.random.shuffle(val_indices)
        valid_indices = val_indices[val_split:]
        test_indices = val_indices[:val_split]
        
        self.train = SubsetRandomSampler(train_indices)
        self.valid = SubsetRandomSampler(valid_indices)
        self.test = SubsetRandomSampler(test_indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


    
def get_datasets(data, target_col, text_col, 
                 collate_func, batch_size,
                 split=0.2, random_seed=42):
    '''
    
    takes a pandas dataframe and returns
    train/valid/test split dataloader objects
    
    Inputs:
        data: pandas dataframe
        target_col: string, name of labels
        text_col: string, name of body text
        collate_func: string, "bow" or "cbow"
        batch_size: int, the number of rows per training batch
        split: float, how much to hold aside for testing and validation
            The percent given is equally split between these.
            The remaining is left for training.
        random_seed: int, can be set to ensure same seq of data
    
    Returns: truple: (train_dataloader, valid_dataloader, test_dataloader)
        all are pyTorch dataloader objects
        
    '''

    data_object = ProjectDataset(data, target_col, text_col,
                                 split, random_seed)
    
    
    
    if collate_func == "cbow":
        print("downloading GloVe, please wait.")
        glove = GloVe(name='6B') #Takes long to download
        cn_func = collate_into_cbow

    else:
        vocab = get_vocab([data_object[i] for i in data_object.train])
        cn_func = collate_into_bow
    

    train_dataloader = DataLoader(data_object, batch_size=batch_size,
                                  sampler=data_object.train, 
                                  collate_fn=cn_func)
    
    valid_dataloader = DataLoader(data_object, batch_size=batch_size,
                                  sampler=data_object.valid, 
                                  collate_fn=cn_func)
    
    test_dataloader = DataLoader(data_object, batch_size=batch_size,
                                  sampler=data_object.test, 
                                  collate_fn=cn_func)

    print("training size: ", len(train_dataloader)*batch_size)
    print("validation size: ", len(valid_dataloader)*batch_size)
    print("testing size: ", len(test_dataloader)*batch_size)
    
    return (train_dataloader, valid_dataloader, test_dataloader)



def get_vocab(training_data):
    
    tokenizer = get_tokenizer('basic_english')
    
    counter = Counter()
    for (label, line) in training_data:
        counter.update(line)
    vocab = Vocab(counter, min_freq=1000)
    return vocab


def collate_into_bow(batch):  
    labels = []
    bag_vector = torch.zeros((len(batch),len(vocab)))
    for i, (label, line) in enumerate(batch):
        labels.append(label-1)
        for w in line:            
            bag_vector[i, vocab[w]] += 1
    
    bag_vector = (bag_vector/bag_vector.sum(axis=1, keepdim=True))
    return torch.tensor(labels), bag_vector
    

def collate_into_cbow(batch):
    cbag_vector = torch.tensor([])
    labels = []
    for i, (label, line) in enumerate(batch):
        labels.append(label-1)
        vecs = glove.get_vecs_by_tokens(line)
        vecs = vecs.sum(axis=0)/vecs.shape[0]
        cbag_vector = torch.cat([cbag_vector, vecs.view(1, -1)])
    
    return torch.tensor(labels), cbag_vector