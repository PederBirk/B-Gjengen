import dataLoader as dl

symbols = ['0','1','2','3','4','5','6','7','8','9']
nTrainingData = 10000
nTestData = 1000
pathToData = 'C:\\Users\\t_tor\\Unsynced\\extracted_images\\'

trainingData,testData = dl.loadData(symbols, nTrainingData, nTestData, pathToData)
