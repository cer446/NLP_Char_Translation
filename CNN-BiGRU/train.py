#from CharAE_Cho import * 

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
from loader import loader, ValidLoader
import numpy as np 

from autoencoder import CharLevel_autoencoder
import pickle 
import sys 

num_epochs = 100
batch_size = 64
learning_rate = 1e-3
max_batch_len = 300
num_symbols = 125
use_cuda = torch.cuda.is_available()

criterion = nn.CrossEntropyLoss()

model = CharLevel_autoencoder(criterion, num_symbols, use_cuda)#, seq_len)
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

def train(model, optimizer, num_epochs, batch_size, learning_rate):
    model.load_state_dict(torch.load('./300_60_teachers6.pth', map_location=lambda storage, loc: storage))
    train_loader, _ = loader()
    valid_loader = ValidLoader()

    for epoch in range(num_epochs):
        model.train()
        for index, (data, label) in enumerate(train_loader):
            data = Variable(data)
            if use_cuda: 
                data = data.cuda()
            
            # ===================forward=====================
            encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
            #print(encoder_outputs.data.shape, encoder_hidden.data.shape) 
                
            decoder_hidden = encoder_hidden
            #print('deocder input', decoder_input.shape, 'decoder hidden', decoder_hidden.data.shape)
            
            loss = model.decode(
                 label, decoder_hidden, encoder_outputs, i =  False)
            
            if index % 1 == 0:
                print(epoch, index, loss.data[0])
                print( evaluate(model, valid_loader, index))
                model.train()

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # ===================log========================
        torch.save(model.state_dict(), './300_60_teachers5.pth')
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.data[0]))

def inference_mode(model, optimizer, num_epochs, batch_size, learning_rate):
    state_path = './300_60_teachers6.pth'
    model.load_state_dict(torch.load(state_path, map_location=lambda storage, loc: storage))
    
    print(state_path)
    valid_loader = ValidLoader()
    model.eval()
    for index, (data, label) in enumerate(valid_loader):
        pickle.dump(label.numpy(), open( "./data/%s_target.p" %(index), "wb" ), protocol=4 )

        data = Variable(data, volatile = True)
        encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
        loss = model.decode(label, encoder_hidden, encoder_outputs, index)
        print(loss.data[0])
         #print(evaluate(model, valid_loader, index))

def evaluate(model, valid_loader, i ):
    model.eval()
    (data, label) = next(iter(valid_loader))
    #pickle.dump(data.numpy(), open( "./data/%s_source.p" %(i), "wb" ), protocol=4 )
    #pickle.dump(label.numpy(), open( "./data/%s_target.p" %(i), "wb" ), protocol=4 )

    data = Variable(data, volatile = True)        
    if use_cuda: 
        data = data.cuda()
    
    encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
        
    loss = model.decode(label, encoder_hidden, encoder_outputs, i)
    return loss.data[0]

if __name__ == '__main__':
    #sys.stdout=open('progress_update_big_data.txt','w')
    #train(model, optimizer, num_epochs, batch_size, learning_rate)
    inference_mode(model, optimizer, num_epochs, batch_size, learning_rate)


    