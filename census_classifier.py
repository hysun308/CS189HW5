#from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn import preprocessing


def segmenter(data, column, value):
    split_function = None;
    split_function = lambda row: row[column] >= value
    set1 = np.array([row for row in data if split_function(row)])
    set2 = np.array([row for row in data if not split_function(row)])
    return (set1, set2)

def uniquecounts(data):
   labels=data[:,data.shape[1]-1]
   results={}
   freq = scipy.stats.itemfreq(labels)
   ct = 0
   for i in freq[:,0]:
       i = int(i)
       results[i]=freq[ct,1]
       ct+=1
   return results

def entropy(data):
   results=uniquecounts(data)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=results[r]/data.shape[0]
      ent=ent-p*np.log2(p)
   return ent

class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb

def buildtree(data,scoref=entropy):
    N = data.shape[0]
    d = data.shape[1]
    if N == 0: return decisionnode()  # len(rows) is the number of units in a set
    current_score = scoref(data)

    # Set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = d-1  # count the # of attributes/columns.
    # It's -1 because the last one is the target attribute and it does not count.
    for col in range(0, column_count):
        # Generate the list of all possible different values in the considered column
        #global column_values  # Added for debugging
        #column_values = {}
        #for row in data:
            #column_values[row[col]] = 1
            # Now try dividing the rows up for each value in this column
        for value in range(int(data[:,col].min()),int(data[:,col].max()+1)):  # the 'values' here are the keys of the dictionnary
            (set1, set2) = segmenter(data, col, value)  # define set1 and set2 as the 2 children set of a division
            if set1.size == 0 or set2.size==0:
                continue

            # Information gain
            n1 = set1.shape[0]
            n2 = set2.shape[0]
            p = set1.shape[0] / N  # p is the size of a child set relative to its parent
            info_gain = current_score -  (n1*scoref(set1) +n2*scoref(set2))/(n1+n2)  # cf. formula information gain
            if info_gain > best_gain and set1.shape[0] > 0 and set2.shape[0] > 0:  # set must not be empty
                best_gain = info_gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # Create the sub branches
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0])
        falseBranch = buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(data))

def csv_out(label):
    with open('heyi.csv','w') as file:
        fwriter = csv.writer(file)
        fwriter.writerow(['Id','Category'])
        for i in range(len(label)):
            fwriter.writerow([i+1,int(label[i])])


def printtree(tree,indent=''):
   # Is this a leaf node?
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->', end=" ")
        printtree(tree.tb,indent+'  ')
        print(indent+'F->', end=" ")
        printtree(tree.fb,indent+'  ')
def classify(data,tree):
        if tree.results != None:
            return (max(tree.results,key=tree.results.get))
        else:
            v = data[tree.col]
            branch = None
            #if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
            #else:
            #    if v == tree.value:
            #        branch = tree.tb
            #    else:
            #       branch = tree.fb
            return classify(data, branch)



def load_data(filename):
    data = pd.read_csv(filename, header=0)
    for col in data:
        print(col)
    return data

if __name__ == "__main__":
    data = load_data("train_data.csv")
    #print(data)
    v = DictVectorizer(sparse=False)
    Data = v.fit_transform(data)
    #print(Data)



    '''
    print("Decision Tree")
    #print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    #csv_out(pred_labels_test)
    tree = buildtree(data_train)
    print(labels_train.shape)
    l = []
    for i in range(N):
        result = classify(data_train[i, 0:32], tree)
        l.append(result)
    pred_labels_train = np.array([l])
    pred_labels_train = pred_labels_train.T
    print(labels_train.shape)
    print(pred_labels_train.shape)
    print("Decision Tree")
    print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_labels_train)))
    t = []
    for i in range(X_test.shape[0]):
        test_result = classify(X_test[i], tree)
        t.append(test_result)
    pred_labels_test = np.array([t])
    pred_labels_test = pred_labels_test.T
    csv_out(pred_labels_test)
    '''