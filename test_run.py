import os
import MARS_train_test as mars
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import pdb


class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input_sequence):
        lstm_out, _ = self.lstm(input_sequence.view(input_sequence.shape[0], 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(input_sequence.shape[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


video_path = '/Users/matthewlevine/Downloads/'
train_videos = [os.path.join('TRAIN_lite_small',v) for v in os.listdir(video_path+'TRAIN_lite_small')]
test_videos = [os.path.join('TEST_lite_small',v) for v in os.listdir(video_path+'TEST_lite_small')]


behs = {}
# behs = {'sniff_face':    ['sniffface', 'snifface', 'sniff-face', 'sniff_face', 'head-investigation','facesniffing']}

ver=[7,8]
verbose=1
feat_type = 'top'
do_wnd = False
do_cwt = False

# behs['sniff_genital'] = ['sniffurogenital','sniffgenitals','sniff_genitals','sniff-genital','sniff_genital',
#                               'anogen-investigation']
# behs['sniff_body'] = ['sniff_body','sniffbody','bodysniffing','body-investigation','socialgrooming',
                          # 'sniff-body','closeinvestigate','closeinvestigation','investigation']

behs['sniff'] = ['sniffface', 'snifface', 'sniff-face', 'sniff_face', 'head-investigation','facesniffing',
                      'sniffurogenital','sniffgenitals','sniff_genitals','sniff-genital','sniff_genital',
                      'anogen-investigation','sniff_body', 'sniffbody', 'bodysniffing', 'body-investigation',
                      'socialgrooming','sniff-body', 'closeinvestigate', 'closeinvestigation', 'investigation']

behs['mount'] = ['mount','aggressivemount','intromission','dom_mount']
behs ['attack'] = ['attack']


X, y, key_order, names = mars.load_data(video_path, test_videos, behs,
                                  ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)

n_features = 22
hidden_dim = 9
X = [x[:,:n_features] for x in X]
num_classes = y[0].shape[1]
input_dim = X[0].shape[1]
model = LSTMTagger(input_dim, hidden_dim, num_classes)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = torch.FloatTensor(X[0])
    tag_scores = model(inputs)
    print(tag_scores)

num_videos = len(X)
num_frames = 1000
num_epochs = 3
for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
	# setence is our features, tags are INDICES of true label
	for v in range(num_videos):
		big_input = X[v]
		big_target = y[v]
		# sample a random chunk of video
		max_start_ind = big_input.shape[0] - num_frames + 1

		start_ind = np.random.randint(max_start_ind)
		end_ind = start_ind + num_frames

		input_sequence = big_input[start_ind:end_ind,:]
		target_sequence = big_target[start_ind:end_ind,:]
		target_inds = torch.tensor(np.argmax(target_sequence, axis=1))

		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		model.zero_grad()

		# Step 2. Get our inputs ready for the network, that is, turn them into
		# Tensors of word indices.

		# Step 3. Run our forward pass.
		tag_scores = model(torch.FloatTensor(input_sequence))

		# Step 4. Compute the loss, gradients, and update the parameters by
		#  calling optimizer.step()
		loss = loss_function(tag_scores, target_inds)
		loss.backward()
		optimizer.step()


# See what the scores are after training
with torch.no_grad():
    inputs = torch.FloatTensor(X[0])
    tag_scores = model(inputs)
    print(tag_scores)


