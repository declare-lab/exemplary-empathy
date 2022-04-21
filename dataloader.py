import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
import hashlib
import os
import pickle
from tqdm import tqdm

class EmpatheticDialogues(Dataset):
    def __init__(self, filename, cache='./.cache'):
        if cache:
            cs = hashlib.md5(open(filename, 'rb').read()).hexdigest()
            cache_file = f'{cache}/{cs}'
            if os.path.isfile(cache_file):
                with open(cache_file, 'rb') as f:
                    self.data = pickle.load(f)
                print(f'Loaded data from {filename} cache')
                return
        data = pd.read_csv(filename, quoting=0).drop(columns=['history'])
        conversations = data['conv_id'].unique().tolist()
        self.data = []
        print(f'Loading data from {filename}')
        for conv_id in tqdm(conversations):
            conv = data.query(f'conv_id == "{conv_id}"').sort_values('utterance_idx')
            context = []
            for idx, utterance in enumerate(conv.iterrows()):
                utterance = utterance[1]
                curr_utterance = utterance['utterance'].replace('_comma_', ',')
                if idx % 2 == 1:
                    self.data.append({
                        'conv_id': conv_id,
                        'emotion': utterance['context'],
                        'context': context[:idx+1],
                        'response': curr_utterance,
                        'exemplars': utterance['exemplars_empd_reddit'].split('ææ'),
                        'sentiment':utterance['sentiment']
                        })
                context.append(curr_utterance)
        if cache:
            if not os.path.isdir(cache):
                os.mkdir(cache)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]

def data_loader(filename, batch_size, shuffle, cache='./.cache'):
    dataset = EmpatheticDialogues(filename, cache=cache)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader
