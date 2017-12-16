import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, src_vocab_size):
        super(Encoder, self).__init__()
        # Initialize source embedding
        self.embedding = nn.Embedding(src_vocab_size, emb_size)
        layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, False))
        self.layers = nn.Sequential(*layers)
                                          
    def forward(self, inputs, input_len):
        # input_len: [batch_size] Variable(torch.LongTensor)
        # h: [batch_size x length x emb_size]
        h = self.embedding(inputs)

        cell_states, hidden_states = [], []
        for layer in self.layers:
            c, h = layer(h)  # c, h: [batch_size x length x hidden_size]            
            time = Variable(torch.arange(0, h.size(1)).unsqueeze(-1).expand_as(h).long())
            if h.is_cuda:
                time = time.cuda()
            # mask to support variable seq lengths
            mask = (input_len.unsqueeze(-1).unsqueeze(-1) > time).float()
            h = h * mask

            # c_last, h_last: [batch_size, hidden_size]           
            c_last = c[range(len(inputs)), (input_len-1).data,:]
            h_last = h[range(len(inputs)), (input_len-1).data,:]
            cell_states.append(c_last)
            hidden_states.append((h_last, h))

        # return lists of cell states and hidden states of each layer
        return cell_states, hidden_states


class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, emb_size, tgt_vocab_size):
        super(Decoder, self).__init__()
        # Initialize target embedding
        self.embedding = nn.Embedding(tgt_vocab_size, emb_size)
        layers = []
        for layer_idx in xrange(n_layers):
            input_size = emb_size if layer_idx == 0 else hidden_size
            use_attn = True if layer_idx == n_layers-1 else False
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, use_attn))
        self.layers = nn.Sequential(*layers)
                                          
    def forward(self, inputs, init_states, memories):
        assert len(self.layers) == len(memories)

        cell_states, hidden_states = [], []
        # h: [batch_size, length, emb_size]
        h = self.embedding(inputs)
        for layer_idx, layer in enumerate(self.layers):
            state = None if init_states is None else init_states[layer_idx]
            memory = memories[layer_idx]

            c, h = layer(h, state, memory)
            cell_states.append(c); hidden_states.append(h)

        # The shape of the each state: [batch_size x length x hidden_size]
        # return lists of cell states and hidden_states
        return cell_states, hidden_states


class QRNNModel(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size, hidden_size,
                 emb_size, src_vocab_size, tgt_vocab_size):
        super(QRNNModel, self).__init__()

        self.encoder = Encoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               emb_size, src_vocab_size)
        self.decoder = Decoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               emb_size, tgt_vocab_size)
        self.proj_linear = nn.Linear(hidden_size, tgt_vocab_size)

    def encode(self, inputs, input_len):
        return self.encoder(inputs, input_len)

    def decode(self, inputs, init_states, memories):
        cell_states, hidden_states = self.decoder(inputs, init_states, memories)
        # return:
        # projected hidden_state of the last layer: logit
        #   first reshape it to [batch_size * length x hidden_size]
        #   after projection: [batch_size * length x tgt_vocab_size]
        h_last = hidden_states[-1]

        return cell_states, self.proj_linear(h_last.view(-1, h_last.size(2)))

    def forward(self, enc_inputs, enc_len, dec_inputs):
        # Encode source inputs
        init_states, memories = self.encode(enc_inputs, enc_len)
        
        # logits: [batch_size * length x tgt_vocab_size]
        _, logits = self.decode(dec_inputs, init_states, memories)

        return logits
