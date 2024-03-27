import pandas as pd
from alex_lair import CKnn, Knn
from typing import Union
from math import floor
import numpy as np
import threading
import matplotlib.pyplot as plt

if __name__ != "__main__":
    exit()

dimensions = (1920, 1080)
kIntervals = 10
totalTests = 5

class test:
    def __init__(self, kValue: int, features: list[str], label: str, baseSet: pd.DataFrame, testSet: pd.DataFrame) -> None:
        self.kValue: int = kValue
        self.features = features
        self.label = label
        self.baseSet = baseSet
        self.testSet = testSet
        self.ncpercentage = 0
        self.cpercentage = 0
        self.testKnn()
        self.testCKnn()
        print("K-Value:", self.kValue)
        print("KNN Percentage:", self.ncpercentage)
        print("CKNN Percentage:", self.cpercentage)
        pass

    def testCKnn(self):
        tempCknn = CKnn(self.kValue)
        self.cpercentage = self.testRun(tempCknn)
        pass

    def testKnn(self):
        tempknn = Knn(self.kValue)
        self.ncpercentage = self.testRun(tempknn)
        pass

    def testRun(self, tempKnn: Union[CKnn, Knn]):
        tempKnn.feed_data(data = self.baseSet)
        totalCorrect = 0
        for i, testValue in enumerate(self.testSet[self.features].values):
            if tempKnn.predict(testValue) == self.testSet[self.label].values[i]:
                totalCorrect += 1
        return totalCorrect / self.testSet.__len__()


class dataset:
    
    def __init__(self, name, features, label, indexCol: int = None,) -> None:
        global totalTests
        df = pd.read_csv(name, index_col=indexCol)
        self.kTests: list[test] = []
        self.features = features
        self.label = label
        self.data = pd.concat((df[features], df[label]), axis=1)
    
    def runDataset(self):
        testRange = self.data.__len__()/totalTests
        for i in range(totalTests):
            testSet = self.data[floor(i*testRange):floor(((i + 1)*testRange))]
            baseSet = np.concatenate((self.data[0:floor(i*testRange)],
                                      (self.data[floor((i + 1)*testRange):self.data.__len__()])))
            tempTest = test(kValue=kIntervals * (i + 1), features=self.features, 
                            label=self.label, baseSet=baseSet, testSet=testSet)
            self.kTests.append(tempTest)

    def getResult(self, testNum: int):
        return self.kTests[testNum]

datasetList: list[dataset] = [
    dataset("./datasets/climber_df.csv", ["height", "weight"], "sex"),
    dataset("./datasets/country_wise_latest.csv", ["Confirmed","Deaths","Recovered","Active","New cases","New deaths","New recovered","Confirmed last week","1 week change", "1 week % increase"], "WHO Region", indexCol="Country/Region"),
    dataset("./datasets/Covid Data.csv", ["USMER","MEDICAL_UNIT","SEX","PATIENT_TYPE","INTUBED","PNEUMONIA","AGE","PREGNANT","DIABETES","COPD","ASTHMA","INMSUPR","HIPERTENSION","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO","ICU"], "CLASIFFICATION_FINAL"),
    dataset("./datasets/iris dataset.csv", ["sep_len","sep_wid","pet_len","pet_wid"], "class"),
    dataset("./datasets/StudentsPerformance.csv", ["math score","reading score","writing score"], "gender")
]

threads: list[threading.Thread] = []

for i in datasetList:
    tempThread = threading.Thread(target=i.runDataset, daemon=True)
    tempThread.start()
    print("Launched thread")
    threads.append(tempThread)

for i in threads:
    i.join()

results = {}

for j in range(totalTests):
    for i in datasetList:
        tempDict = results.get(str(kIntervals * (j + 1)), {"knn": 0, "cknn": 0})
        results[str(kIntervals * (j + 1))] = {"knn": tempDict["knn"] + i.getResult(j).ncpercentage, "cknn": tempDict["cknn"] + i.getResult(j).cpercentage}
    tempDict = results.get(str(kIntervals * (j + 1)))
    results[str(kIntervals * (j + 1))] = {"knn": round((tempDict["knn"] / len(datasetList)) * 100), "cknn": round((tempDict["cknn"] / len(datasetList)) * 100)}

kValuesList = list(results.keys())
kresults = {
    "knn Results": [],
    "cknn Results": []
}

for _, key in enumerate(results):
    kresults["knn Results"].append(results[key]["knn"])
    kresults["cknn Results"].append(results[key]["cknn"])

x = np.arange(len(kValuesList))
width = 0.25
multiplier = 0.5
fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in kresults.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

ax.set_ylabel('Percentage of correct predictions')
ax.set_xticks(x + width, kValuesList)
ax.legend(loc='best', ncols=2)
ax.set_ylim(0, 100)

plt.show()
