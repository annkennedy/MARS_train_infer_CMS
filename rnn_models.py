import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, bidirectional, num_layers):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        num_directions = 2 if bidirectional else 1

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional = bidirectional, num_layers=num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(num_directions*hidden_dim, num_classes)

    def forward(self, input_sequence, bias_sequence=0):
        # optional bias term driven by glm_scores
        lstm_out, _ = self.lstm(input_sequence.view(input_sequence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(input_sequence.shape[0], -1))
        predicted_class_scores = bias_sequence + F.log_softmax(tag_space, dim=1)
        return predicted_class_scores

class GRUTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, bidirectional, num_layers):
        super(GRUTagger, self).__init__()
        self.hidden_dim = hidden_dim

        num_directions = 2 if bidirectional else 1

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional = bidirectional, num_layers=num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(num_directions*hidden_dim, num_classes)

    def forward(self, input_sequence, bias_sequence=0):
        # optional bias term driven by glm_scores
        gru_out, _ = self.gru(input_sequence.view(input_sequence.shape[0], 1, -1))
        tag_space = self.hidden2tag(gru_out.view(input_sequence.shape[0], -1))
        pdb.set_trace()
        predicted_class_scores = bias_sequence + F.log_softmax(tag_space, dim=1)
        return predicted_class_scores


def get_optimizer(name, params, lr=None):
    if name=='SGD':
        if lr is None:
            lr = 0.1
        return optim.SGD(params, lr=lr)
    elif name=='Adam':
        if lr is None:
            lr = 0.01
        return optim.Adam(params, lr=lr)
    elif name=='LBFGS':
        if lr is None:
            lr = 1
        return optim.LBFGS(params, lr=lr)
    elif name=='RMSprop':
        if lr is None:
            lr = 0.01
        return optim.RMSprop(params, lr=lr)
    else:
        return None

def get_loss(name, weight=None):
    if name=='nn.NLLLoss':
        return nn.NLLLoss(weight=weight)
    else:
        return None

def get_model(name, input_dim, hidden_dim, num_classes, bidirectional = False, num_layers=1):
    if name=='LSTMTagger':
        return LSTMTagger(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, bidirectional=bidirectional, num_layers=num_layers)
    elif name=='GRUTagger':
        return GRUTagger(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, bidirectional=bidirectional, num_layers=num_layers)
    else:
        return None
