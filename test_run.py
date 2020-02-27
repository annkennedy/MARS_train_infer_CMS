import os
import MARS_train_test as mars
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import *
from rnn_models import *
import pdb

import argparse


parser = argparse.ArgumentParser(description='behaviorRNN')
parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--num_frames', type=int, default=1000, help='number of frames per training video chunk')
parser.add_argument('--hidden_dim', type=int, default=10, help='number of dimensions for RNN hidden state')
parser.add_argument('--optimizer', type=str, default='SGD', help='specifiy which optimizer to use')
parser.add_argument('--loss', type=str, default='nn.NLLLoss', help='specifiy which loss function to use')
parser.add_argument('--model_name', type=str, default='LSTMTagger', help='specifiy which RNN model to use')
parser.add_argument('--train_path', type=str, default='/Users/matthewlevine/Downloads/TRAIN_lite_small', help='specifiy path to TRAIN videos')
parser.add_argument('--test_path', type=str, default='/Users/matthewlevine/Downloads/TEST_lite_small', help='specifiy path to TEST videos')
parser.add_argument('--balance_weights', type=str2bool, default=True, help='If true, compute cost function weights based on relative class frequencies')
FLAGS = parser.parse_args()


def main():

	train_video_path, train_video_nm = os.path.split(FLAGS.train_path)
	test_video_path, test_video_nm = os.path.split(FLAGS.test_path)

	train_videos = [os.path.join(train_video_nm,v) for v in os.listdir(FLAGS.train_path)]
	test_videos = [os.path.join(test_video_nm,v) for v in os.listdir(FLAGS.test_path)]


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
	Xtrain, ytrain, key_order_train, names_train = mars.load_data(train_video_path, train_videos, behs,
	                                  ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)

	Xtest, ytest, key_order_test, names_test = mars.load_data(test_video_path, test_videos, behs,
	                                  ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)


	n_features = 22
	Xtrain = [x[:,:n_features] for x in Xtrain]
	Xtest = [x[:,:n_features] for x in Xtest]
	num_classes = ytrain[0].shape[1]
	input_dim = Xtrain[0].shape[1]
	# model = LSTMTagger(input_dim=input_dim, hidden_dim=FLAGS.hidden_dim, num_classes=num_classes)
	model = get_model(name=FLAGS.model_name, input_dim=input_dim, hidden_dim=FLAGS.hidden_dim, num_classes=num_classes)

	optimizer = get_optimizer(name=FLAGS.optimizer, params=model.parameters(), lr=FLAGS.lr)
	# optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr)

	## Normalize the data
	Xtrain_stats = stats_of(Xtrain)
	Xtrain = normalize(X=Xtrain, stats=Xtrain_stats)
	Xtest = normalize(X=Xtest, stats=Xtrain_stats) # using Xtrain stats on purpose here...for now.


	## Report on class balance of Train and Test sets
	foo = np.concatenate(ytrain).sum(axis=0)
	train_fracs = foo / float(foo.sum())
	print('Train Class Balance:', train_fracs)
	foo = np.concatenate(ytest).sum(axis=0)
	test_fracs = foo / float(foo.sum())
	print('Test Class Balance:', test_fracs)

	# compute sample weight based on inverse frequencies
	# https://stats.stackexchange.com/questions/342170/how-to-train-an-lstm-when-the-sequence-has-imbalanced-classes
	if FLAGS.balance_weights:
		weight = 1. / np.concatenate(ytrain).sum(axis=0)
		weight = weight / weight.sum()
		weight = torch.FloatTensor(weight)
	else:
		weight = None

	loss_function = get_loss(name=FLAGS.loss, weight=weight) #e.g. nn.NLLLoss()

	# train the model
	num_frames = FLAGS.num_frames
	num_epochs = FLAGS.num_epochs
	for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
		# setence is our features, tags are INDICES of true label
		all_predicted_classes = []
		all_predicted_scores = []
		all_targets = []
		for v in range(len(Xtrain)):
			big_input = Xtrain[v]
			big_target = ytrain[v]

			offset = np.random.randint(len(big_input) % num_frames - 1)
			permutations = list(range(int(len(big_input) / num_frames)))
			np.random.shuffle(permutations)
			for permutation in permutations:
				# sample a random chunk of video
				start_ind = permutation * num_frames + offset
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
		train_recall = recall(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
		train_precision = precision(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
		print('Epoch',epoch,' Train Loss=', train_loss.data.numpy().item())
		print('Epoch',epoch,' Train Recall=', train_recall)
		print('Epoch',epoch,' Train Precision=', train_precision)

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
		test_recall = recall(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
		test_precision = precision(predicted=all_predicted_classes.data.numpy(), actual=all_targets.data.numpy())
		print('Epoch',epoch,' Test Loss=', test_loss.data.numpy().item())
		print('Epoch',epoch,' Test Recall=', test_recall)
		print('Epoch',epoch,' Test Precision=', test_precision)


if __name__ == '__main__':
	main()


