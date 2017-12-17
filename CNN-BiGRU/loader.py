''' loads pickles produced via preprocessing.py
    instantiates a pytorch loader according to input parameters
'''
from torch.utils.data import TensorDataset,Dataset, DataLoader
#from util.preprocessing import *
import numpy as np
import pickle
import torch 

def ValidLoader(batch_size = 64, shuffle = False, path = './data/validation/'):
    source = pickle.load(open(path + 'valid.p', "rb"))
    target = pickle.load(open(path + "valid_target.p", "rb" ))
    source, target = np.array(source), np.array(target)
    valid_dataset = TensorDataset(torch.from_numpy(source), 
    torch.from_numpy(target)) 

    valid_loader = DataLoader(
    valid_dataset, batch_size = batch_size, shuffle = shuffle, drop_last=True)

    return valid_loader

def loader( batch_size = 64, shuffle = True, train_portion = 0.99, seq_format = False):
    path = './data/'
    names = [str(i) for i in range(10000, 210000, 10000)]

    source_data, target_data = False, False
    
    for number in names:
        target_chunk = pickle.load(open(path + '%sl.p' %number, "rb"))
        source_chunk = pickle.load(open(path + '%s.p' %number, "rb"))
        
        if  type(source_data) == bool:
            source_data, target_data = source_chunk, target_chunk
            continue
        #print( number, target_chunk.shape, source_chunk.shape)        
        #print(target_chunk.shape, source_chunk.shape)

        source_data = np.vstack( (source_data, source_chunk ) )
        target_data = np.vstack( ( target_data, target_chunk ) )

    source_data = np.array(source_data)
    #print(source_data.shape)
    target_data = np.array(target_data)

    #dataset = pickle.load(open(data_pickle_path, "rb"))
    num_samples = source_data.shape[0]
    #print(num_samples)
    uniform_sampling = np.random.random_sample((num_samples,))

    train_dataset = source_data[ uniform_sampling < train_portion]
    #print(train_dataset.shape)
    train_labels = target_data[  uniform_sampling < train_portion]
    #print(train_labels.shape, train_dataset.dtype)
    valid_dataset = source_data[ uniform_sampling >= train_portion ]
    valid_labels = target_data[  uniform_sampling >= train_portion]
    #print(train_dataset.shape, type(train_dataset))

    #print(len(train_dataset[0]), len(train_dataset[4]))
    source = TensorDataset(torch.from_numpy(train_dataset), 
    torch.from_numpy(train_labels)) 

    train_loader = DataLoader(
    source, batch_size = batch_size, shuffle = shuffle, drop_last=True)

    valid_source = TensorDataset(torch.from_numpy(valid_dataset),
    torch.from_numpy(valid_labels)) 

    valid_loader = DataLoader(
    valid_source, batch_size = batch_size, shuffle = shuffle, drop_last=True)
    
    return train_loader, valid_loader

if __name__ == '__main__':
   pass