import os
from sklearn.metrics import accuracy_score

eval_dir = './output/MR'
y_true = []
y_pred = []
with open(os.path.join(eval_dir, 'label_test.txt'), 'r') as f:
	lines = f.readlines()
	for line in lines:
		t, p = line.strip().split()
		y_true.append(t)
		y_pred.append(p)

print(accuracy_score(y_true, y_pred))