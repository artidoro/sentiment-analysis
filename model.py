import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import utils
from itertools import chain

"""
Module for Multinomial Naive Bayes Classifier MNBC.
This implements the model described by Wang and Manning at: 
http://www.aclweb.org/anthology/P/P12/P12-2.pdf#page=118
"""

class MNBC(nn.Module):

	def __init__(self, vocab_size, alpha):
		super(MNBC, self).__init__()
		self.vocab_size = vocab_size
		self.alpha = alpha

		self.w = nn.Embedding(vocab_size, 1)
		self.b = Variable(torch.FloatTensor(1).zero_())

	def forward(self, batch):
		dim1, dim2 = batch.size()
		emb_prod = self.w(batch.view(1,-1).squeeze()).view(dim1, dim2)
		dot_prod = torch.sum(emb_prod, 1)
		linear_separator = dot_prod + self.b
		pos = torch.sigmoid(linear_separator)
		neg = torch.sigmoid(-linear_separator)
		return torch.cat((pos.view(-1,1), neg.view(-1,1)), 1)

	def predict(self, batch):
		probs = self.forward(batch.text)
		_, argmax = (probs.max(1))
		return argmax + 1

	# training method for module takes the training iterator (in place)
	def train(self, train_iter, val_iter, test_iter):
    	# initialize the counters
		p = Variable(torch.FloatTensor(self.vocab_size).zero_()) + self.alpha
		q = Variable(torch.FloatTensor(self.vocab_size).zero_()) + self.alpha

    	# keep track of positive and negative examples
		N_positive = 0
		N_negative = 0
		for batch in tqdm(train_iter):
    		# update variables p,q, N, N
			N_negative += torch.sum(batch.label - 1).data.numpy()
			N_positive += torch.sum(batch.label % 2).data.numpy()
			label = batch.label.data
			
			for i, s in enumerate(batch.text):
				if label[i] == 1:
					p[s] += 1
				else:
					q[s] += 1

		self.w.weight = torch.nn.Parameter(torch.log((p / torch.sum(p)) / (q / torch.sum(q))).data)
		self.b = torch.log(Variable(torch.FloatTensor(N_positive / N_negative)))

		filename = "mnbc/pred" + ".txt"
		print ("Outputing predictions")
		utils.test_model(self, test_iter, filename)
		return 

"""
Logistic regression model
"""

class LogReg(nn.Module):
	def __init__(self, vocab_size):
		super(LogReg, self).__init__()
		self.vocab_size = vocab_size

		self.w = nn.Embedding(vocab_size, 1)
		self.b = Variable(torch.FloatTensor(1).zero_(), requires_grad=True)

		self.loss = nn.BCEWithLogitsLoss()
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

	def forward(self, batch):
		emb_prod = self.w(batch)
		dot_prod = torch.sum(emb_prod, 1)
		linear_separator = dot_prod + self.b
		return linear_separator.squeeze()

	def predict(self, batch):
		scores = self.forward(batch.text)
		preds = (scores >= 0).type(torch.LongTensor)
		return preds + 1

	# training method for module takes the training iterator (in place)
	def train(self, train_iter, val_iter, num_epochs=2000):
		for epoch in range(num_epochs):
			total_loss = 0
			train_len = 0

			for batch in tqdm(chain(train_iter, val_iter)):
				batch.text.volatile = False
				batch.label.volatile = False

				self.optimizer.zero_grad()
				probs = self.forward(batch.text)
				ys = (batch.label - 1).type(torch.FloatTensor)
				output = self.loss(probs, ys)
				output.backward()
				self.optimizer.step()
				total_loss += output.data[0]
				train_len += 1
			total_loss /= train_len

			if not epoch % 20:
				val_loss, val_accuracy = utils.calc_accuracy_1dim(self, val_iter)
				print('Epoch {}/{}: training loss is {}; val loss is {}'.format(epoch, num_epochs, total_loss, val_loss))
				print('val accuracy is {}'.format(val_accuracy))

"""
Continuous Bag of Words model

Pre-trained word embeddings are summed to obtain a representation of the 
sentence and a linear layer is applyied to the result

"""

class CBOW(nn.Module):

	def __init__(self, vocab_size, pretrained_embed, vect_size):
		super(CBOW, self).__init__()
		self.vocab_size = vocab_size
		self.pretrained_embed = pretrained_embed
		self.vect_size = vect_size

		self.embed = nn.Embedding(vocab_size, vect_size)
		self.embed.weight = nn.Parameter(pretrained_embed)
		self.w = nn.Linear(vect_size, 1)

		self.loss = nn.BCEWithLogitsLoss()
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

	def forward(self, batch):
		emb_prod = self.embed(batch)
		cont_bow = torch.sum(emb_prod, 1)
		linear_separator = self.w(cont_bow)
		return linear_separator.squeeze()

	def predict(self, batch):
		scores = self.forward(batch.text)
		preds = (scores >= 0).type(torch.LongTensor)
		return preds + 1

	def train(self, train_iter, val_iter, num_epochs=50):
		for epoch in range(num_epochs):
			total_loss = 0
			train_len = 0
			# initialize the counters
			for batch in tqdm(chain(train_iter, val_iter)):
				batch.text.volatile = False
				batch.label.volatile = False

				self.optimizer.zero_grad()
				scores = self.forward(batch.text)
				ys = (batch.label - 1).type(torch.FloatTensor)
				output = self.loss(scores, ys)
				output.backward()
				self.optimizer.step()

				total_loss += output.data[0]
				train_len += 1
				total_loss /= train_len

			if not epoch % 5:
				val_loss, val_accuracy = utils.calc_accuracy_1dim(self, val_iter)
				print('Epoch {}/{}: training loss is {}; val loss is {}'.format(epoch, num_epochs, total_loss, val_loss))
				print('val accuracy is {}'.format(val_accuracy))

"""
2 channels Convolutional Neural Netowrk 
as described by Kim in the paper https://arxiv.org/pdf/1408.5882.pdf
pretrained word embeddings, 3 stride sizes for convolution layers, 
ReLu activation and max polling, drop out regularization, normalization
of the linear fully connected layer  
"""

class CNN(nn.Module):

	def __init__(self, embeddings):
		super(CNN, self).__init__()
		self.vocab_size = embeddings.size(0)
		self.embed_dim = embeddings.size(1)

		self.w = nn.Embedding(self.vocab_size, self.embed_dim)
		self.w.weight = nn.Parameter(embeddings)

		self.w_static = nn.Embedding(self.vocab_size, self.embed_dim)
		self.w_static.weight = nn.Parameter(embeddings, requires_grad=False)

		self.conv1 = nn.Conv1d(300, 100, 3, padding=1, stride=1)
		self.conv2 = nn.Conv1d(300, 100, 4, padding=2, stride=1)
		self.conv3 = nn.Conv1d(300, 100, 5, padding=2, stride=1)

		self.conv4 = nn.Conv1d(300, 100, 3, padding=1, stride=1)
		self.conv5 = nn.Conv1d(300, 100, 4, padding=2, stride=1)
		self.conv6 = nn.Conv1d(300, 100, 5, padding=2, stride=1)

		self.fc1 = nn.Linear(600, 1)
		self.loss = nn.BCEWithLogitsLoss()

		self.optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, self.parameters()))


	def forward(self, batch, training=False):
		x_dynamic = self.w(batch).transpose(1,2)
		x_static = self.w_static(batch).transpose(1,2)

		c1 = F.relu(self.conv1(x_static))
		c2 = F.relu(self.conv2(x_static))
		c3 = F.relu(self.conv3(x_static))
		c4 = F.relu(self.conv4(x_dynamic))
		c5 = F.relu(self.conv5(x_dynamic))
		c6 = F.relu(self.conv6(x_dynamic))

		z1 = F.max_pool1d(c1, c1.size(2)).view(batch.size(0), -1)
		z2 = F.max_pool1d(c2, c2.size(2)).view(batch.size(0), -1)
		z3 = F.max_pool1d(c3, c3.size(2)).view(batch.size(0), -1)
		z4 = F.max_pool1d(c4, c4.size(2)).view(batch.size(0), -1)
		z5 = F.max_pool1d(c5, c5.size(2)).view(batch.size(0), -1)
		z6 = F.max_pool1d(c6, c6.size(2)).view(batch.size(0), -1)

		z = torch.cat((z1, z2, z3, z4, z5, z6), dim=1)

		d = F.dropout(z, 0.5, training)
		y = self.fc1(d).squeeze()
		return y

	def predict(self, batch):
		scores = self.forward(batch.text)
		preds = (scores >= 0).type(torch.LongTensor)
		return preds + 1

	def train(self, train_iter, val_iter, test_iter, num_epochs):
		for epoch in tqdm(range(num_epochs)):

			if not epoch % 1:
				val_accuracy = utils.calc_accuracy(self, val_iter)
				train_accuracy = utils.calc_accuracy(self, train_iter)
				print ("Epoch: " + str(epoch))
				print ("Train Accuracy: " + str(train_accuracy))
				print ("Validation Accuracy: " + str(val_accuracy))

			if not epoch % 1:
				filename = "cnn/pred" + str(epoch) + ".txt"
				print ("Outputing predictions")
				utils.test_model(self, test_iter, filename)

			for batch in tqdm(chain(train_iter, val_iter)):
				batch.text.volatile = False
				batch.label.volatile = False
				self.optimizer.zero_grad()
				probs = self.forward(batch.text, training=True)
				ys = (batch.label - 1).type(torch.FloatTensor)
				output = self.loss(probs, ys)				
				output.backward()
				self.optimizer.step()

				# Regularize by capping fc layer weights at norm 3
				if torch.norm(self.fc1.weight.data) > 3.0:
					self.fc1.weight = nn.Parameter(3.0 * self.fc1.weight.data / torch.norm(self.fc1.weight.data))

"""
Like the previous by 2 layer CNN
"""

class CNN2(nn.Module):

	def __init__(self, embeddings):
		super(CNN2, self).__init__()
		self.vocab_size = embeddings.size(0)
		self.embed_dim = embeddings.size(1)

		self.w = nn.Embedding(self.vocab_size, self.embed_dim)
		self.w.weight = nn.Parameter(embeddings)

		self.w_static = nn.Embedding(self.vocab_size, self.embed_dim)
		self.w_static.weight = nn.Parameter(embeddings, requires_grad=False)

		nc1 = 32
		nc2 = 64


		self.conv1 = nn.Conv1d(300, nc1, 3, padding=1, stride=1)
		self.conv2 = nn.Conv1d(300, nc1, 4, padding=2, stride=1)
		self.conv3 = nn.Conv1d(300, nc1, 5, padding=2, stride=1)
		self.conv4 = nn.Conv1d(300, nc1, 3, padding=1, stride=1)
		self.conv5 = nn.Conv1d(300, nc1, 4, padding=2, stride=1)
		self.conv6 = nn.Conv1d(300, nc1, 5, padding=2, stride=1)

		self.conv21 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
		self.conv22 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
		self.conv23 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
		self.conv24 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
		self.conv25 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
		self.conv26 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)

		self.fc1 = nn.Linear(64 * 6, 1)
		self.loss = nn.BCEWithLogitsLoss()

		self.optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, self.parameters()))


	def forward(self, batch, training=False):
		x_dynamic = self.w(batch).transpose(1,2)
		x_static = self.w_static(batch).transpose(1,2)

		c1 = F.relu(self.conv1(x_dynamic))
		c2 = F.relu(self.conv2(x_dynamic))
		c3 = F.relu(self.conv3(x_dynamic))
		c4 = F.relu(self.conv4(x_static))
		c5 = F.relu(self.conv5(x_static))
		c6 = F.relu(self.conv6(x_static))

		z1 = F.max_pool1d(c1, 3, padding=1)
		z2 = F.max_pool1d(c2, 3, padding=1)
		z3 = F.max_pool1d(c3, 3, padding=1)
		z4 = F.max_pool1d(c4, 3, padding=1)
		z5 = F.max_pool1d(c5, 3, padding=1)
		z6 = F.max_pool1d(c6, 3, padding=1)

		c21 = F.relu(self.conv21(z1))
		c22 = F.relu(self.conv22(z2))
		c23 = F.relu(self.conv23(z3))
		c24 = F.relu(self.conv24(z4))
		c25 = F.relu(self.conv25(z5))
		c26 = F.relu(self.conv26(z6))

		z21 = F.max_pool1d(c21, c21.size(2)).view(batch.size(0), -1)
		z22 = F.max_pool1d(c22, c22.size(2)).view(batch.size(0), -1)
		z23 = F.max_pool1d(c23, c23.size(2)).view(batch.size(0), -1)
		z24 = F.max_pool1d(c24, c24.size(2)).view(batch.size(0), -1)
		z25 = F.max_pool1d(c25, c25.size(2)).view(batch.size(0), -1)
		z26 = F.max_pool1d(c26, c26.size(2)).view(batch.size(0), -1)

		z = torch.cat((z21, z22, z23, z24, z25, z26), dim=1)

		d = F.dropout(z, 0.5, training)
		y = self.fc1(d).squeeze()
		return y

	def predict(self, batch):
		scores = self.forward(batch.text)
		preds = (scores >= 0).type(torch.LongTensor)
		return preds + 1

	def train(self, train_iter, val_iter, test_iter, num_epochs):
		for epoch in tqdm(range(num_epochs)):

			if not epoch % 1:
				val_accuracy = utils.calc_accuracy(self, val_iter)
				train_accuracy = utils.calc_accuracy(self, train_iter)
				print ("Epoch: " + str(epoch))
				print ("Train Accuracy: " + str(train_accuracy))
				print ("Validation Accuracy: " + str(val_accuracy))

			if not epoch % 1:
				filename = "magic/pred" + str(epoch) + ".txt"
				print ("Outputing predictions")
				utils.test_model(self, test_iter, filename)

			for batch in tqdm(chain(train_iter, val_iter)):
				batch.text.volatile = False
				batch.label.volatile = False
				self.optimizer.zero_grad()
				probs = self.forward(batch.text, training=True)
				ys = (batch.label - 1).type(torch.FloatTensor)
				output = self.loss(probs, ys)				
				output.backward()
				self.optimizer.step()

				# Regularize by capping fc layer weights at norm 3
				if torch.norm(self.fc1.weight.data) > 3.0:
					self.fc1.weight = nn.Parameter(3.0 * self.fc1.weight.data / torch.norm(self.fc1.weight.data))

"""
LSTM model

Bidirectional LSTM on top of pretrained word embeddings.
100 dimensional hidden representation output per direction, so 200 dimension
final hidden representation. Relu and Max pooling, and linear layer with
regularization. 

Takes number of layers as input parameter (we tested 1 and 2 layers)
"""

class LSTM(nn.Module):

	def __init__(self, embeddings, layers=1):
		super(LSTM, self).__init__()
		self.vocab_size = embeddings.size(0)
		self.embed_dim = embeddings.size(1)

		self.w = nn.Embedding(self.vocab_size, self.embed_dim)
		self.w.weight = nn.Parameter(embeddings)

		# biderectional LSTM layer
		self.lstm = nn.LSTM(300, 100, layers, batch_first=True, dropout=0.5, bidirectional=True)
		self.c0 = Variable(torch.FloatTensor(layers * 2, 1, 100).zero_(), requires_grad=True)
		self.h0 = Variable(torch.FloatTensor(layers * 2, 1, 100).zero_(), requires_grad=True)

		self.fc1 = nn.Linear(200, 1)
		self.loss = nn.BCEWithLogitsLoss()

		self.optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, self.parameters()))


	def forward(self, batch, training=False):
		h = torch.cat([self.h0 for _ in range(batch.size(0))], 1)
		c = torch.cat([self.c0 for _ in range(batch.size(0))], 1)

		H, hn = self.lstm(self.w(batch), (h, c))
		a = F.relu(H)
		a = a.transpose(1,2)
		z = F.max_pool1d(a, a.size(2)).view(-1, 200)

		d = F.dropout(z, 0.5, training)
		
		y = self.fc1(d).squeeze()
		return y

	def predict(self, batch):
		scores = self.forward(batch.text)
		preds = (scores >= 0).type(torch.LongTensor)
		return preds + 1

	def train(self, train_iter, val_iter, test_iter, num_epochs):
		for epoch in tqdm(range(num_epochs)):

			# if not epoch % 1:
			# 	train_accuracy = utils.calc_accuracy(self, train_iter)
			# 	print ("Epoch: " + str(epoch))
			# 	print ("Train Accuracy: " + str(train_accuracy))
			# 	print ("Validation Accuracy: " + str(val_accuracy))

			if not epoch % 1:
				filename = "lstm/pred" + str(epoch) + ".txt"
				print ("Outputing predictions")
				utils.test_model(self, test_iter, filename)

			for batch in (chain(train_iter, val_iter)):
				batch.text.volatile = False
				batch.label.volatile = False
				self.optimizer.zero_grad()
				probs = self.forward(batch.text, training=True)
				ys = (batch.label - 1).type(torch.FloatTensor)
				output = self.loss(probs, ys)				
				output.backward()
				self.optimizer.step()

				# Regularize by capping fc layer weights at norm 3
				if torch.norm(self.fc1.weight.data) > 3.0:
					self.fc1.weight = nn.Parameter(3.0 * self.fc1.weight.data / torch.norm(self.fc1.weight.data))

"""
LSTM - CNN model

We apply a CNN as described earlier to the LSTM output

"""

class LSTMCNN(nn.Module):

	def __init__(self, embeddings):
		super(LSTMCNN, self).__init__()
		self.vocab_size = embeddings.size(0)
		self.embed_dim = embeddings.size(1)

		self.w = nn.Embedding(self.vocab_size, self.embed_dim)
		self.w.weight = nn.Parameter(embeddings)

		# biderectional LSTM layer
		self.lstm = nn.LSTM(300, 64, 1, batch_first=True, dropout=0.5, bidirectional=True)
		self.c0 = Variable(torch.FloatTensor(2, 1, 64).zero_(), requires_grad=True)
		self.h0 = Variable(torch.FloatTensor(2, 1, 64).zero_(), requires_grad=True)

		# cnn
		self.conv1 = nn.Conv1d(128, 64, 3, padding=1, stride=1)
		self.conv2 = nn.Conv1d(128, 64, 4, padding=2, stride=1)
		self.conv3 = nn.Conv1d(128, 64, 5, padding=2, stride=1)


		self.fc1 = nn.Linear(192, 1)
		self.loss = nn.BCEWithLogitsLoss()

		self.optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, self.parameters()))


	def forward(self, batch, training=False):
		h = torch.cat([self.h0 for _ in range(batch.size(0))], 1)
		c = torch.cat([self.c0 for _ in range(batch.size(0))], 1)

		H, hn = self.lstm(self.w(batch), (h, c))
		a = F.relu(H)

		a = a.transpose(1,2)

		conv1 = F.relu(self.conv1(a))
		conv2 = F.relu(self.conv2(a))
		conv3 = F.relu(self.conv3(a))

		z1 = F.max_pool1d(conv1, conv1.size(2)).view(batch.size(0), -1)
		z2 = F.max_pool1d(conv2, conv2.size(2)).view(batch.size(0), -1)
		z3 = F.max_pool1d(conv3, conv3.size(2)).view(batch.size(0), -1)
		
		z = torch.cat((z1, z2, z3), dim=1)

		d = F.dropout(z, 0.5, training)
		y = self.fc1(d).squeeze()

		return y

	def predict(self, batch):
		scores = self.forward(batch.text)
		preds = (scores >= 0).type(torch.LongTensor)
		return preds + 1

	def train(self, train_iter, val_iter, test_iter, num_epochs):
		for epoch in tqdm(range(num_epochs)):

			# if not epoch % 1:
			# 	# val_accuracy = utils.calc_accuracy(self, val_iter)
			# 	train_accuracy = utils.calc_accuracy(self, train_iter)
			# 	print ("Epoch: " + str(epoch))
			# 	print ("Train Accuracy: " + str(train_accuracy))
			# 	# print ("Validation Accuracy: " + str(val_accuracy))

			if not epoch % 1:
				filename = "lstmcnn/pred" + str(epoch) + ".txt"
				print ("Outputing predictions")
				utils.test_model(self, test_iter, filename)

			for batch in tqdm(chain(train_iter, val_iter)):
				batch.text.volatile = False
				batch.label.volatile = False
				self.optimizer.zero_grad()
				probs = self.forward(batch.text, training=True)
				ys = (batch.label - 1).type(torch.FloatTensor)
				output = self.loss(probs, ys)				
				output.backward()
				self.optimizer.step()

				# Regularize by capping fc layer weights at norm 3
				if torch.norm(self.fc1.weight.data) > 3.0:
					self.fc1.weight = nn.Parameter(3.0 * self.fc1.weight.data / torch.norm(self.fc1.weight.data))


					