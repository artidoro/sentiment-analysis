import utils
import numpy as np 
import torch

fns = ["ensemble2/cnnpred5.txt", "ensemble2/nbpred.txt", "ensemble2/lstm85.txt"]

def get_data(fn):
	with open(fn, 'r') as file:
		lines = file.readlines()
		data = list(map(lambda l: int(l.split()[0].split(',')[1]), lines[1:]))
		return data

l = []

for fn in fns:
	data = get_data(fn)
	l.append(data)

l = np.array(l) - 1

res = np.array(list(map(lambda l: 1 if sum(l) >= 2 else 0, l.T))) + 1

batch_size = 10
train_iter, val_iter, test_iter, TEXT = utils.torchtext_extract(batch_size)

ans = torch.LongTensor()
for batch in test_iter:
	ans = torch.cat((ans, batch.label.data), 0)

print(float(sum(res == ans.numpy())) / len(res))

filename = "ensemble3.txt"
with open(filename, "w") as f:
    f.write("Id,Cat\n")
    for i, u in enumerate(res):
        f.write(str(i) + "," + str(u) + "\n")