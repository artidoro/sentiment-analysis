import torchtext
from torchtext.vocab import Vectors, GloVe
from torch import FloatTensor, LongTensor


def torchtext_extract(batch_size=10):

	TEXT = torchtext.data.Field(batch_first=True)
	LABEL = torchtext.data.Field(sequential=False)


	train, val, test = torchtext.datasets.SST.splits(
	    TEXT, LABEL,
	    filter_pred=lambda ex: ex.label != 'neutral')

	TEXT.build_vocab(train)
	LABEL.build_vocab(train)

	train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
	    (train, val, test), batch_size=batch_size, repeat=False, device=-1)

	#url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
	#TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

	TEXT.vocab.load_vectors(vectors=GloVe(name='6B'))

	return train_iter, val_iter, test_iter, TEXT


def calc_accuracy(model, test_iter):
    correct = 0
    total = 0

    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        preds = model.predict(batch)

        correct += sum(preds == batch.label).data.numpy()[0]
        total += batch.text.size()[0]
    return float(correct) / total


def calc_accuracy_1dim(model, test_iter):  # TODO: consolidate 2 versions
    correct = 0.
    total = 0.
    loss = 0.

    for batch in test_iter:
        scores = model(batch.text)
        preds = (scores >= 0).type(LongTensor)
        ys = (batch.label - 1).type(FloatTensor)
        output = model.loss(scores, ys)

        correct += sum(preds == ys.type(LongTensor)).data[0]
        loss += output.data[0]
        total += batch.text.size()[0]
    loss = loss / total * len(test_iter)
    accuracy = correct / total
    return loss, accuracy


def test_model(model, test_iter, filename):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    
    correct = 0
    total = 0

    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        preds = model.predict(batch)
        upload += list(preds.data.numpy())

        correct += sum(preds == batch.label).data.numpy()[0]
        total += batch.text.size()[0]

    print("Test Accuracy: " + str(float(correct) / total))

    with open(filename, "w") as f:
        f.write("Id,Cat\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")



