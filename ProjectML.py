import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold, neighbors, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from scipy.spatial import distance


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
            self.features = np.array(data)

            self.label = [labels[i] for i in label]
            self.name = name
        else:
            self.features = []
            self.label = []
            self.name = []

        self.classes = 6
        self.n_files = 0


def knnClassifier(data,labels, nNeighbors=20):
    knn = neighbors.KNeighborsClassifier(nNeighbors, weights='distance')
    knn.fit(data,labels)
    return knn

def svmClassifier(data,trainLabels):
    svm = LinearSVC()
    svm.fit(data,trainLabels)
    return svm

def testClass(classifier, classData, testData, testLabels):

    prediction = classifier.predict(testData)
    accuracy = metrics.accuracy_score(testLabels, prediction)
    print("\n Accuracy: %f",accuracy)
    print("\n Results from classifier:  \n %s \n"
          % ( metrics.classification_report(testLabels, prediction)))
    print("\n Confussion matrix:\n %s" % metrics.confusion_matrix(testLabels, prediction))
    kfold = 10
    scores = cross_val_score(classifier, classData.features, classData.label, cv=kfold)
    print("\n Cross validation score: \n")
    print(scores)
    print("\n Mean cv score: %f", np.mean(scores))

def holdOut(dataClass, percentSplit):
    trainData, testData, trainLabels, expectedLabels = train_test_split(dataClass.features, dataClass.label,
                                                                       test_size=(1.0 - percentSplit), random_state=0)
    return trainData,testData,trainLabels,expectedLabels

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
    print(files)
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

trainData, testData, trainLabels, expectedLabels = holdOut(dataClass, 0.6)
knn = knnClassifier(trainData,trainLabels,10)
svm = svmClassifier(trainData,trainLabels)
print("\n Report for K-NN classifier: \n")
testClass(knn, dataClass,testData,expectedLabels)
print("\n Report for SVM classifier: \n")
testClass(svm, dataClass,testData,expectedLabels)





