#################################################################
# Natalie Leon Guerrero 
# CS 445 Machine Learning
# Homework 1 - Perceptrons
# Perceptron class
# Code taken and adapted from stephanmonika.net
#################################################################

import numpy as np
import csv

'''Perceptron Class'''
class pcn:

    def __init__(self,trainData,outputs):
        self.trainData = trainData
        self.nIn = (np.shape(trainData)[1])    		# num of columns/x values
        self.nSets = np.shape(trainData)[0]	        # num of rows/examples
        self.nOut = np.shape(outputs)[0]		# num of outputs/targets
        # Randomize weights from -0.05 to 0.05
        self.weights = np.random.rand(self.nIn,self.nOut)*0.1-0.05


    def train(self,testData,options,eta,nIterations):
        self.weights = np.array(self.weights)
        accuracyArr = np.zeros(nIterations)

        nSetsTest = np.shape(testData)[0]		# num of rows/examples

        # Randomize training and testing data
        np.random.shuffle(self.trainData)
        np.random.shuffle(testData)

        # Separate training targets from training inputs
        targetsTrain = trainData[:,0]	
        targetsTrain = np.vstack(targetsTrain)
        inputsTrain = trainData[:,1:]
        inputsTrain = inputsTrain/255

        # Separate testing targets from testing inputs
        targetsTest = testData[:,0]	
        targetsTest = np.vstack(targetsTest)
        inputsTest = testData[:,1:]
        inputsTest = inputsTest/255

        # Add bias to inputs
        inputsTrain = np.concatenate((np.ones((self.nSets,1)),inputsTrain),axis=1)
        inputsTest = np.concatenate((np.ones((nSetsTest,1)),inputsTest),axis=1)

        print "========================================="
        print "Training set, Eta =", eta
        print "========================================="

        with open('pcn.csv','a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Eta"] + [eta])
            writer.writerow(["Epoch"] + ["Accuracy Training"] + ["Accuracy Testing"])

        for it in range(nIterations):
            accurateTrain = 0
            accurateTest = 0

            # Train Model
            for idx in range(self.nSets):
                prediction, outputs = self.predict(inputsTrain[idx])
                if prediction != targetsTrain[idx]:
                    # change the weights
                    y = self.activationY(outputs)
                    t = self.activationT(targetsTrain[idx])
                    self.weights -= eta*np.transpose(np.dot(np.vstack(y-t),np.asmatrix(inputsTrain[idx])))
                else:
                    accurateTrain += 1

            # Test Model
            for idx in range(nSetsTest):
                prediction, outputs = self.predict(inputsTest[idx])
                if prediction == targetsTest[idx]:
                    accurateTest += 1

            # Note the accuracy of the epoch
            accuracyArr[it] = (float(accurateTrain)/float(self.nSets))*100
            accuracyTest = (float(accurateTest)/float(nSetsTest)*100)
            print "Epoch: ", it, "    Accuracy Training: %", int(accuracyArr[it]), "    Accuracy Testing: %", int(accuracyTest)

            with open('pcn.csv','a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([it] + [accuracyArr[it]] + [accuracyTest])

            if accuracyArr[int(it-1)] > (accuracyArr[it] + 1):
                break
        print
        print "--------- DONE WITH TRAINING ----------"
        print
        return


    def predict(self,inputs):
        outputs = np.dot(inputs,self.weights)
        prediction = np.argmax(outputs)
        return prediction, outputs 

    def activationY(self, outputs):		
        return np.where(outputs>0,1,0)

    def activationT(self, target):
        t = np.zeros(self.nOut)
        t[int(target)] = 1
        return t

    def confuseMatrix(self, testData):
        if(np.shape(testData)[1] != self.nIn):
            print "Data given does not have the same number of inputs."
            print "Cannot test data"
            return

        nSets = np.shape(testData)[0]
        np.random.shuffle(testData)
        targets = testData[:,0]
        targets = np.vstack(targets)
        inputs = testData[:,1:]
        inputs = inputs/255
        inputs = np.concatenate((np.ones((nSets,1)),inputs),axis=1)

        confuseM = np.zeros((self.nOut,self.nOut))
        accurate = 0

        for idx in range(nSets):
            prediction, outputs = self.predict(inputs[idx])
            if prediction == targets[idx]:
                accurate += 1
            confuseM[int(prediction)][int(targets[idx])] += 1

        accuracy = int((float(accurate)/float(nSets))*100)
        print "Resulting Accuracy: %", accuracy
        np.set_printoptions(suppress=True)
        print "Confusion Matrix"
        print confuseM
        print

        with open('pcn.csv','a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Confusion Matrix"])
            for i in range(self.nOut):
                writer.writerow(confuseM[i,:])
            writer.writerow(["Resulting Accuracy"] + [accuracy])
            writer.writerow([])

        return

# ================================================================================

def readFile(filename):
    data = list(csv.reader(open(filename), delimiter=','))
    data = np.array(data).astype("float")
    return data

if __name__ == '__main__':
    # test example
    #inputs, targets = readFile("example.csv")
    #outputs = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8]])

    trainData = readFile("mnist_train.csv")
    testData = readFile("mnist_test.csv")
    outputs = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])

    if (trainData.size > 0) and (testData.size > 0) and (outputs.size > 0):
        with open('pcn.csv','w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Perceptron Training"])
            writer.writerow(["EXAMPLE 1"])
        p1 = pcn(trainData,outputs)
        p1.train(testData,outputs,0.1,70)
        p1.confuseMatrix(testData)

        with open('pcn.csv','a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["EXAMPLE 2"])
        p2 = pcn(trainData,outputs)
        p2.train(testData,outputs,0.01,70)
        p2.confuseMatrix(testData)

        with open('pcn.csv','a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["EXAMPLE 3"])
        p3 = pcn(trainData,outputs)
        p3.train(testData,outputs,0.001,70)
        p3.confuseMatrix(testData)
	