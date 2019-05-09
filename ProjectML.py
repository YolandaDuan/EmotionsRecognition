import os
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn import manifold, neighbors, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC

from scipy.spatial import distance

'''
Global Variables 
'''
SIMPLE_EMBEDDING=False
MANUAL_SPLIT    =True
DATA_2D         =False

labels ={
    "ANGER": 0,
    "DISGUST": 1,
    "FEAR": 2,
    "HAPPY": 3,
    "SADNESS": 4,
    "SURPRISE": 5,
    }

class classData:
    def __init__(self,dir=None):
        if dir:
            data,label,name = readLm3(dir)
            print(np.shape(data))
            print(np.shape(label))
            print(np.shape(name))
            self.features = np.array(data)

            self.label = [labels[i] for i in label]
            self.name = name
        else:
            self.features = []
            self.label = []
            self.name = []

        self.classes = 6
        self.n_files = 0


def knnClassifier(data,labels, nNeighbors):
    knn = neighbors.KNeighborsClassifier(nNeighbors, weights='uniform')
    knn.fit(data,labels)
    return knn

def svmClassifier(data,trainLabels):
    svm = LinearSVC()
    svm.fit(data,trainLabels)
    return svm

def testClass(classifier, classData, testData, testLabels):

    prediction = classifier.predict(testData)
    accuracy = metrics.accuracy_score(testLabels, prediction)
    print("Accuracy: ",accuracy)
    print("Results from classifier:  \n %s \n"
          % ( metrics.classification_report(testLabels, prediction)))
    print("Confussion matrix:\n %s" % metrics.confusion_matrix(testLabels, prediction))
    kfold = 5
    scores = cross_val_score(classifier, classData.features, classData.label, cv=kfold)
    print("Cross validation score: ")
    print(scores)
    print("Mean cv score: ", np.mean(scores))

""" def holdOut(dataClass, percentSplit):
    trainData, testData, trainLabels, expectedLabels = train_test_split(dataClass.features, dataClass.label,
                                                                       test_size=(1.0 - percentSplit), random_state=0)
    return trainData,testData,trainLabels,expectedLabels """

def holdOut(dataClass, nSamples, percentSplit=0.6):
    '''
    This function splits the data into training and test sets
    '''
    if(MANUAL_SPLIT):
        n_trainSamples = int(nSamples * percentSplit)
        trainData = dataClass.features[:n_trainSamples]
        trainLabels = dataClass.label[:n_trainSamples]
        
        testData = dataClass.features[n_trainSamples:]
        expectedLabels = dataClass.label[n_trainSamples:]
    else:
        trainData, testData, trainLabels, expectedLabels = train_test_split(dataClass.features, dataClass.label,
                                                                            test_size=(1.0-percentSplit), random_state=0)

    return trainData,testData,trainLabels,expectedLabels

def plotData(X, labelsData):
    '''
    This function plots either 2D data or 3D data
    '''
    if(DATA_2D):
        # Convert labels from string into integer, then plot it in different colors
        labelsData = pd.Categorical(pd.factorize(labelsData)[0])
        plt.scatter(X[:,0], X[:,1], c=labelsData, cmap=plt.cm.summer)
        plt.show()
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X[:,0], X[:,1], X[:,2])
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.show()

# Read Files
def readLm3(directory):
    currLoc = os.getcwd()
    path = currLoc +"/dataBosphoruslm3"
    files = []
    data = []
    labels = []
    names = []

    for file in os.listdir(path):
        if file.endswith(".lm3"):
            files.append(str(file))
    # print(files)
    for file in files:
        lm3Data = parseLm3File(path+"/"+file)
        if lm3Data:
            data.append(lm3Data)
            labels.append(file.split("_")[2])
            names.append(file)

    return data, labels, names

def parseLm3File(filePath):
    """
    Parse the lm3 file at filePath and return the 3D landmarks.
    Args:
        filePath(string): A string of the filepath to the lm3 file.
    Returns:
        array: A array of the 3D landmarks.
    """
    landmarkNames = {
        "Outer left eyebrow",
        "Middle left eyebrow",
        "Inner left eyebrow",
        "Inner right eyebrow",
        "Middle right eyebrow",
        "Outer right eyebrow",
        "Outer left eye corner",
        "Inner left eye corner",
        "Inner right eye corner",
        "Outer right eye corner",
        "Nose saddle left",
        "Nose saddle right",
        "Left nose peak",
        "Nose tip",
        "Right nose peak",
        "Left mouth corner",
        "Upper lip outer middle",
        "Right mouth corner",
        "Upper lip inner middle",
        "Lower lip inner middle",
        "Lower lip outer middle",
        "Chin middle",
    }

    with open(filePath) as f:
        lines = f.readlines()

    landmarks = []

    for i in range(4, len(lines), 2):
        if lines[i - 1].rstrip() in landmarkNames:
            landmark = [float(j) for j in lines[i].split()]
            landmarks.append(landmark)

    nlandmarks = len(landmarks)
    if nlandmarks != 22:
        return False

    return landmarks


path = "dataBosphoruslm3"

dataClass = classData(path)
dataClass.features = dataClass.features.reshape(dataClass.features.shape[0], -1)
n_samples = len(dataClass.features)
print(dataClass)
print(dataClass.features)
print(dataClass.features.shape)

# Data Reduction 
if(SIMPLE_EMBEDDING):
    #  Dimensionality Reduction PCA 
    pca = PCA(n_components = 3)
    X_trans = pca.fit_transform(dataClass.features)
else:
    # Manifold embedding with tSNE/Users/Lingyan/Desktop/IIS2019/Assignment 1/emotions_recognition.py
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    X_trans = tsne.fit_transform(dataClass.features)

# Ploting Data
plotData(X_trans, dataClass.label)

trainData, testData, trainLabels, expectedLabels = holdOut(dataClass, n_samples)
knn = knnClassifier(trainData,trainLabels,50)
svm = svmClassifier(trainData,trainLabels)
print("\nReport for K-NN classifier\n")
testClass(knn, dataClass,testData,expectedLabels)
print("\nReport for SVM classifier\n")
testClass(svm, dataClass,testData,expectedLabels)





