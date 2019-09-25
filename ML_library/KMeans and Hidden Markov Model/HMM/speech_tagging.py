import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	
	
	# num_sentence = len(train_data)
	# num_tags =  len(tags)
	# pi = np.zeros((num_tags))
	# train_data = np.array(train_data)
	# first_tags = np.array([x.tags[0] for x in train_data])
	# for i in range(num_tags):
	# 	pi[i] = np.array(np.where(first_tags == tags[i])).size / num_sentence
	
	# A = np.zeros((num_tags, num_tags))

	# #a[i, j] ===> i to j
	# for i in range(num_tags):
	# 	for j in range(num_tags):
	# 		#num of transitions starting from i
	# 		n = 0
	# 		for k in range(num_sentence):
	# 			n += train_data[k].tags[:-1].count(tags[i])
			
			
	# 		#num of transitions from i to j
	# 		m = 0
	# 		for k in range(num_sentence):
	# 			for l in range(len(train_data[k].tags) - 1):
	# 				if train_data[k].tags[l] == tags[i] and train_data[k].tags[l + 1] == tags[j]:
	# 					m += 1
	# 		if n == 0:
	# 			A[i, j] = 0
	# 		else:
	# 			A[i, j] = m / n
	

	# state_to_symbol = [[]] * num_tags
	# for sentence in train_data:
	# 	for i in range(num_tags):
	# 		for j in range(len(sentence.tags)):
	# 			if tags[i] == sentence.tags[j]:
	# 				state_to_symbol[i].append(sentence.words[j])
	# symbols, counts = np.unique(state_to_symbol, return_counts=True)
	# num_symbols = len(symbols)
	# B = np.zeros((num_tags, num_symbols))

	# for i in range(num_symbols):
	# 	for j in range(num_tags):
	# 		#num of this symbol i given tag j
	# 		n = 0
	# 		for o in state_to_symbol[j]:
	# 			n += o.count(symbols[i])
	# 		#num of all unique symbols in given tag j
	# 		m = len(np.unique(state_to_symbol[j]))

	# 		B[j, i] = n / m
	# obs_dict = {}
	# for s in symbols:
	# 	obs_dict[s] = np.where(symbols == s)
	# state_dict = {}
	# for i in range(num_tags):
	# 	state_dict[tags[i]] = i


	# model = HMM(pi, A, B, obs_dict, state_dict)
				

	
	

					
			



	###################################################
	return model


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	obs_dict = model.obs_dict
	B = model.B
	num_state, num_obs_symbol = B.shape
	for sentence in test_data:
		for word in sentence.words:
			if word not in obs_dict.keys():
				obs_dict[word] = num_obs_symbol
				extra = np.full((num_state, 1), 10e-6)
				B = np.hstack((B, extra))
	print(B)
	print(obs_dict)



	num_sentence = len(test_data)
	tagging = [[]] * num_sentence
	for i in range(num_sentence):
		Osequence = np.array(test_data[i].words)
		tagging[i] = model.viterbi(Osequence)




	###################################################
	return tagging

