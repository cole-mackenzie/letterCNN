import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd
import random

def cleanLines(line):
    
    splitLine = line.split(',')
    splitLine = [val.strip() for val in splitLine]
    splitLine = [val for val in splitLine if val != '']
    answer = splitLine[0]
    values = splitLine[1:]
    
    #values = [val.replace('\n', '') for val in values]
    #values = [val.replace('C', '') for val in values]
    
    values = np.asarray([((int(val) / 255) * .99) + 0.01 for val in values])
    
    return (answer, values)
	
def to_one_hot(labels, dimension):

    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
		
    return results

def convToArr(arr):
        
    newArr = np.array([int((((val - 0.01) / 0.99) * 255)) for val in arr])
    newArr = newArr.reshape(20, 10, 3)
    
    return newArr

def rowThreshold(image, minLen):
    
    starts = []
    ends = []
    
    for x in range(0, len(image)):
        
        if x < minLen:
            pass
        
        elif image[x, :, :].mean() != 255 and image[max(x - minLen, 0):x, :, :].mean() == 255:
            starts.append(x)
        
        elif image[max(0, x - (minLen - 1)):x + 1, :, :].mean() == 255 and image[max(0, x - minLen), :, :].mean() != 255:
            ends.append(x - minLen + 1)
        
        else:
            pass
    
    if len(starts) > len(ends):
        
        ends.append(starts[-1] + 1)
    
    thresholds = [(starts[y], ends[y]) for y in range(0, len(starts))]
    
    return thresholds
	
	
def columnThreshold(image, userThresh):

    starts = []
    ends = []
    
    for x in range(0, image.shape[1]):
        
        if x == 0:
            pass
        
        elif image[:, x - 1].mean() >= userThresh and image[:, x].mean() < userThresh:
            starts.append(x)
            
        elif image[:, x - 1].mean() < userThresh  and image[:, x].mean() >= userThresh:
            ends.append(x)
            
        else:
            pass
        
    thresholds = [(starts[y], ends[y]) for y in range(0, len(starts))]
    
    return thresholds


def findSpaces(textBlock, userThresh):
    
    starts = []
    ends = []
    
    for x in range(0, textBlock.shape[1]):
        
        if (x == 0 and textBlock[:, x, :].mean() < userThresh) or (textBlock[:, x, :].mean() < userThresh and textBlock[:, x - 1, :].mean() >= userThresh):
            starts.append(x)
        
        elif (textBlock[:, x, :].mean() >= userThresh and textBlock[:, x - 1, :].mean() < userThresh):
        
            ends.append(x)
    
    if len(starts) > len(ends):
        
        ends.append(starts[-1] + 1)
    
    combined = [(starts[y], ends[y]) for y in range(0, len(starts))]
    
    return combined


def padColumns(curArr, rowLen, channelLen):
    
    numCols = random.randint(50, 100)
    
    emptyArr = np.full((rowLen, numCols, channelLen), 255)
    listArr = [emptyArr, curArr]
    random.shuffle(listArr)
    
    return np.concatenate(listArr, axis=1)


def padRows(curArr, numCols, channelLen):
    
    rowLen = random.randint(50, 100)
    
    emptyArr = np.full((rowLen, numCols, channelLen), 255)
    listArr = [emptyArr, curArr]
    random.shuffle(listArr)
    
    return np.concatenate(listArr, axis=0)

	
with open(r'testData.csv', 'r') as f:
    data = f.readlines()


allVals = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "@", "#", "$", "%", "^", "&", "*", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

sampleValues = []
sampleLables = []

for x in range(0, len(data)):
    cleanedRow = cleanLines(data[x])
    sampleLables.append(cleanedRow[0])
    sampleValues.append(cleanedRow[1])


tempValues = [np.array(Image.fromarray(convToArr(val).reshape(20, 10, 3).astype(np.uint8)).resize((100, 200))) for val in sampleValues]

sampleValues = []
for val in tempValues:

    valMask = np.ones(val.shape, dtype="bool")
    valThresh = val < 180
    valMask[valThresh] = False
    val[valMask] = 255
    sampleValues.append(val)
	

mainContainer = []

for k in range(0, 5):
    
    for val, lab in zip(sampleValues, sampleLables):

        tempDict = {}
        dice = random.randint(1, 6)

        if dice == 3:
            tempDict['Label'] = lab
            tempDict['Value'] = val    

        else:
            tempDict = {}
            tempDict['Label'] = lab

            newVal = padColumns(val, val.shape[0], val.shape[2])
            newVal = padRows(newVal, newVal.shape[1], newVal.shape[2])
            newVal = np.asarray(Image.fromarray(newVal.astype(np.uint8)).resize((100, 200)))

            tempDict['Value'] = newVal

        mainContainer.append(tempDict)
	

random.shuffle(mainContainer)

sampleValues = [d['Value'] for d in mainContainer]
sampleLabels = [d['Label'] for d in mainContainer]

model = models.load_model(r'cnnmodels')

val_samp = np.array(sampleValues[:13848])
partial_val_samp = np.array(sampleValues[13848:])

one_hot_train_labels = to_one_hot([allVals.index(val) for val in sampleLabels], len(allVals))

label_samp = one_hot_train_labels[:13848]
partial_label_samp = one_hot_train_labels[13848:]

model.compile(
				optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy']
             )

			 
for circ in range(1):
    
    print('Circuit:', circ)
    
    sampleValues = []
    sampleLables = []

    for x in range(0, len(data)):
        cleanedRow = cleanLines(data[x])
        sampleLables.append(cleanedRow[0])
        sampleValues.append(cleanedRow[1])

    tempValues = [np.array(Image.fromarray(convToArr(val).reshape(20, 10, 3).astype(np.uint8)).resize((100, 200))) for val in sampleValues]
    
    sampleValues = []
    for val in tempValues:

        valMask = np.ones(val.shape, dtype="bool")
        valThresh = val < np.random.randint(150, 255)
        valMask[valThresh] = False
        val[valMask] = 255
        sampleValues.append(val)
    
    mainContainer = []
    
    for k in range(0, 5):
    
        for val, lab in zip(sampleValues, sampleLables):

            tempDict = {}
            dice = random.randint(1, 6)

            if dice == 3:
                tempDict['Label'] = lab
                tempDict['Value'] = val    

            else:
                tempDict = {}
                tempDict['Label'] = lab

                newVal = padColumns(val, val.shape[0], val.shape[2])
                newVal = padRows(newVal, newVal.shape[1], newVal.shape[2])
                newVal = np.asarray(Image.fromarray(newVal.astype(np.uint8)).resize((100, 200)))

                tempDict['Value'] = newVal

            mainContainer.append(tempDict)
    
    random.shuffle(mainContainer)
    sampleValues = [d['Value'] for d in mainContainer]
    sampleLabels = [d['Label'] for d in mainContainer]
    
    #val_samp = np.array(sampleValues[:13848])
    partial_val_samp = np.array(sampleValues)

    one_hot_train_labels = to_one_hot([allVals.index(val) for val in sampleLabels], len(allVals))

    #label_samp = one_hot_train_labels[:13848]
    partial_label_samp = one_hot_train_labels
    
    batches = [1000]
    
    for batch in batches:
        
        print('Batch Num:', batch)
        
        history = model.fit(partial_val_samp,
                              partial_label_samp,
                              epochs=1,
                              batch_size=batch,
                              shuffle=True
                             )

history.save(r'cnnmodels')


