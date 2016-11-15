from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio


class DecisionTree:
    def __init__(self,split_rule,left,right,label):
        self.left = left
        self.right = right
        self.label = label
        self.split_rule = split_rule
    def impurity(self,left_label_hist,right_label_hist):
     nl = left_label_hist.shape[0]
     nr = right_label_hist.shape[0]
     hl = 0
     hr = 0
     numsl = np.sum(left_label_hist[:,1])
     numsr = np.sum(right_label_hist[:,1])
     for i in range(nl):
     	hl-=(left_label_hist[i,1]/numsl)*np.log(left_label_hist[i,1]/numsl)
     for j in range(nr):
     	hr-= (right_label_hist[i,1]/numsr)*np.log(right_label_hist[i,1]/numsr)
     entro = (numsl*hl+numsr*hr)/(numsl+numsr)
     info_gain = (hl+hr)-entro
     return info_gain
    def segmenter(data,column,value):
        split_function = None;
        split_function = lambda row:row[column]>=value
        set1 = np.array([row for row in data if split_function(row)])
        set2 = np.array([row for row in data if not split_function(row)])
        return(set1,set2)
    def train(self,data,labels):
     return
    def predict(self,data):
      return

def csv_out(label):
    with open('heyi1.csv','w') as file:
        fwriter = csv.writer(file)
        fwriter.writerow(['Id','Category'])
        for i in range(len(label)):
            fwriter.writerow([i+1,int(label[i])])

def uniquecounts(data):
   labels=data[:,data.shape[1]-1]
   results={}
   freq = scipy.stats.itemfreq(labels)
   for i in freq[:,0]:
       i = int(i)
       results[i]=freq[i,1]
   return results
def entropy(data):
   results=uniquecounts(data)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=results[r]/data.shape[0]
      ent=ent-p*np.log2(p)
   return ent
if __name__ == "__main__":

    data = sio.loadmat('spam_data.mat')
    X_train = data['training_data']
    X_test = data['test_data']
    labels_train = data['training_labels'] 
    y_train=labels_train
    data_train = np.zeros((X_train.shape[0],X_train.shape[1]+1))
    data_train[:,0:X_train.shape[1]]=X_train
    data_train[:,X_train.shape[1]]=y_train
    print(y_train.max())
    print(y_train.min())
    print(uniquecounts(data_train))
    print(entropy(data_train))


    print("Decision Tree")
    #print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    #csv_out(pred_labels_test)