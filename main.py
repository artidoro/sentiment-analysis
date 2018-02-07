import torchtext
from torchtext.vocab import Vectors, GloVe
import model
import utils

"""
main file to run all the models.

First get the iteratorrs then pass them to the models for training.
The models will output validation results during training.

The models take different parameters (see model.py), they all have a train 
method. 

Note that for the last version of our model we trained on both validation and 
and training data. We stopped the training process after the same number of 
iterations that gave the optimal result when just training on training data.
Alternatively we could have used cross validation on both training and 
validation data. This did not significantly increase our accuracy.

"""
alpha = 1
batch_size = 10
vect_size = 300
num_epochs = 5

train_iter, val_iter, test_iter, TEXT = utils.torchtext_extract(batch_size)

vocab_size = len(TEXT.vocab)

#MNBC = model.MNBC(vocab_size, alpha)
#MNBC.train(train_iter, val_iter, test_iter)

#LogReg = model.LogReg(vocab_size)
#LogReg.train(train_iter, val_iter, num_epochs=2)

#CBOW = model.CBOW(vocab_size, TEXT.vocab.vectors, vect_size)
#CBOW.train(train_iter, val_iter, num_epochs=1)

#CNN = model.CNN(embeddings=TEXT.vocab.vectors)
#CNN.train(train_iter, val_iter, test_iter, num_epochs=200)

#CNN2 = model.CNN2(embeddings=TEXT.vocab.vectors)
#CNN2.train(train_iter, val_iter, test_iter, num_epochs=200)

LSTM = model.LSTM(embeddings=TEXT.vocab.vectors, layers=1)
LSTM.train(train_iter, val_iter, test_iter, num_epochs=5)

# LSTMCNN = model.LSTMCNN(embeddings=TEXT.vocab.vectors)
# LSTMCNN.train(train_iter, val_iter, test_iter, num_epochs=5)