'''
train_test_datasets.py

make train/valid/test splits for evaluation

'''

import pandas as pd
from sklearn.model_selection import train_test_split

MAX_SEQ_LEN = 100

def trim_string(x, max_len=MAX_SEQ_LEN):
    x = x.split(maxsplit=max_len)
    x = ' '.join(x[:max_len])
    return x

def make_data(raw_data_path, destination_folder, test_perc):
    '''
    '''
    
    train_test_ratio = 1.0 - (test_perc / 2)
    train_valid_ratio = 1.0 - (test_perc / 2)
    
    # Read raw data
    df_raw = pd.read_csv(raw_data_path)

    # Prepare columns
    df_raw['label'] = df_raw['vaderCat'].astype('int')
    
    df_raw.loc[df_raw['label'] == 1, 'label'] = 5
    df_raw.loc[df_raw['label'] == 2, 'label'] = 1

    # Drop rows with empty text
    df_raw.drop( df_raw[df_raw.text.isnull()].index, inplace=True)


    # Trim text to first_n_words
    df_raw['text'] = df_raw['text'].apply(trim_string)
    df_raw.drop( df_raw[df_raw.text.str.len() < 5].index, inplace=True)

    df_raw = df_raw[['label', 'text']]

    # Split according to label
    df_low_all = df_raw[df_raw['label'] == 0]
    #df_med_all = df_raw[df_raw['label'] == 1]
    df_high = df_raw[df_raw['label'] == 1]

    # Undersample dataset
    #undersample_med = round(df_high.shape[0]/df_med_all.shape[0], 2)
    undersample_low = round(df_high.shape[0]/df_low_all.shape[0], 2)

    #df_med, df_rejects = train_test_split(df_med_all, train_size=undersample_med, random_state=1)
    df_low, df_rejects = train_test_split(df_low_all, train_size=undersample_low, random_state=1)

    # Train-test split
    df_low_full_train, df_low_test = train_test_split(df_low, train_size = train_test_ratio, random_state = 1)
    #df_med_full_train, df_med_test = train_test_split(df_med, train_size = train_test_ratio, random_state = 1)
    df_high_full_train, df_high_test = train_test_split(df_high, train_size = train_test_ratio, random_state = 1)

    # Train-valid split
    df_low_train, df_low_valid = train_test_split(df_low_full_train, train_size = train_valid_ratio, random_state = 1)
    #df_med_train, df_med_valid = train_test_split(df_med_full_train, train_size = train_valid_ratio, random_state = 1)
    df_high_train, df_high_valid = train_test_split(df_high_full_train, train_size = train_valid_ratio, random_state = 1)

    # Concatenate splits of different labels
    df_train = pd.concat([df_low_train, df_high_train], ignore_index=True, sort=False)
    df_valid = pd.concat([df_low_valid, df_high_valid], ignore_index=True, sort=False)
    df_test = pd.concat([df_low_test, df_high_test], ignore_index=True, sort=False)

    # Write preprocessed data
    df_train.to_csv(destination_folder + '/train.csv', index=False)
    df_valid.to_csv(destination_folder + '/valid.csv', index=False)
    df_test.to_csv(destination_folder + '/test.csv', index=False)

    # Print results
    print("Datasets saved.")
    print(f'{destination_folder}/train.csv saved; {df_train.shape[0]} records')
    print(f'{destination_folder}/valid.csv saved; {df_valid.shape[0]} records')
    print(f'{destination_folder}/test.csv saved; {df_test.shape[0]} records')
    print()
    print("checking class balance:")
    print("low in train: ", df_train[df_train['label']==0].shape[0])
    print("med in train: ", df_train[df_train['label']==1].shape[0])
    print("high in train: ", df_train[df_train['label']==2].shape[0])
    
    return None
