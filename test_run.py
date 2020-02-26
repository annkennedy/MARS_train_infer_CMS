import os
import MARS_train_test as mars
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import *
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
        predicted_class_scores = F.log_softmax(tag_space, dim=1)
        return predicted_class_scores


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


# Read in Train and Test sets
Xtrain, ytrain, key_order_train, names_train = mars.load_data(video_path, train_videos, behs,
                                  ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)

Xtest, ytest, key_order_test, names_test = mars.load_data(video_path, test_videos, behs,
                                  ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)


n_features = 22
hidden_dim = 9
Xtrain = [x[:,:n_features] for x in Xtrain]
Xtest = [x[:,:n_features] for x in Xtest]
num_classes = ytrain[0].shape[1]
input_dim = Xtrain[0].shape[1]
model = LSTMTagger(input_dim, hidden_dim, num_classes)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

## Normalize the data
Xtrain_stats = stats_of(Xtrain)
Xtrain = normalize_maxmin(X=Xtrain, stats=Xtrain_stats)
Xtest = normalize_maxmin(X=Xtest, stats=Xtrain_stats) # using Xtrain stats on purpose here...for now.


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = torch.FloatTensor(Xtest[0])
    predicted_class_scores = model(inputs)
    print(predicted_class_scores)

num_frames = 1000
num_epochs = 10
for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
	# setence is our features, tags are INDICES of true label
	all_predicted_classes = []
	all_predicted_scores = []
	all_targets = []
	for v in range(len(Xtrain)):
		big_input = Xtrain[v]
		big_target = ytrain[v]
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

		# Step 3. Run our forward pass.
		predicted_class_scores = model(torch.FloatTensor(input_sequence))
		predicted_class_index = torch.argmax(predicted_class_scores, axis=1) # this is the model's class prediction i.e. the highest scoring element

		# Step 4. Compute the loss, gradients, and update the parameters by
		#  calling optimizer.step()
		loss = loss_function(predicted_class_scores, target_inds)
		loss.backward()
		optimizer.step()

		all_predicted_classes.append(predicted_class_index)
		all_predicted_scores.append(predicted_class_scores)
		all_targets.append(target_inds)


	all_predicted_classes = torch.cat(all_predicted_classes)
	all_predicted_scores = torch.cat(all_predicted_scores)
	all_targets = torch.cat(all_targets)

	# Report Train losses after each epoch
	train_loss = loss_function(all_predicted_scores, all_targets)
	# train_recall = recall(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
	# train_precision = precision(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
	print('Epoch',epoch,' Train Loss=', train_loss)
	# print('Epoch',epoch,' Train Recall=', train_recall)
	# print('Epoch',epoch,' Train Precision=', train_precision)

	# Report TEST performance after each epoch
	all_predicted_classes = []
	all_predicted_scores = []
	all_targets = []
	for v in range(len(Xtest)):
		big_input = Xtest[v]
		big_target = ytest[v]
		input_sequence = big_input
		target_sequence = big_target
		target_inds = torch.tensor(np.argmax(target_sequence, axis=1))

		# Step 3. Run our forward pass.
		predicted_class_scores = model(torch.FloatTensor(input_sequence))
		predicted_class_index = torch.argmax(predicted_class_scores, axis=1) # this is the model's class prediction i.e. the highest scoring element

		all_predicted_classes.append(predicted_class_index)
		all_predicted_scores.append(predicted_class_scores)
		all_targets.append(target_inds)
	# Step 4. Compute the losses
	all_predicted_classes = torch.cat(all_predicted_classes)
	all_predicted_scores = torch.cat(all_predicted_scores)
	all_targets = torch.cat(all_targets)
	test_loss = loss_function(all_predicted_scores, all_targets)
	# test_recall = recall(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
	# test_precision = precision(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
	print('Epoch',epoch,' Test Loss=', test_loss)
	# print('Epoch',epoch,' Test Recall=', test_recall)
	# print('Epoch',epoch,' Test Precision=', test_precision)


# See what the scores are after training
with torch.no_grad():
    inputs = torch.FloatTensor(Xtest[0])
    predicted_class_scores = model(inputs)
    print(predicted_class_scores)



