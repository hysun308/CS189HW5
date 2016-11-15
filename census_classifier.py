from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy

class DecisionTree:
    def __init__(split_rule,left,right,label):
        self.left = left
        self.right = right
        self.label = label
        self.split_rule = split_rule
    def impurity(left_label_hist,right_label_hist):
     return  
    def segmenter(data,labels):
     return
    def train(data,labels):
     return
    def predict(data):
      return
def csv_out(label):
    with open('heyi1.csv','w') as file:
        fwriter = csv.writer(file)
        fwriter.writerow(['Id','Category'])
        for i in range(len(label)):
            fwriter.writerow([i+1,int(label[i])])
def csv_read(filename):
	with open(filename) as file:
		data = csv.DictReader(file,['age','workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
		for row in data:
			print(row['workclass'])
		return data

if __name__ == "__main__":
    data = csv_read("train_data.csv")
    print(data)

    print("Decision Tree")
    #print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    #csv_out(pred_labels_test)