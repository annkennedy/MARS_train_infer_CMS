import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_sequence):
        lstm_out, _ = self.lstm(input_sequence.view(input_sequence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(input_sequence.shape[0], -1))
        predicted_class_scores = F.log_softmax(tag_space, dim=1)
        return predicted_class_scores

def get_optimizer(name, params, lr):
    if name=='SGD':
        return optim.SGD(params, lr=lr)
    else:
        return None

def get_loss(name, weight=None):
    if name=='nn.NLLLoss':
        return nn.NLLLoss(weight=weight)
    else:
        return None

def get_model(name, input_dim, hidden_dim, num_classes):
    if name=='LSTMTagger':
        return LSTMTagger(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    else:
        return None

