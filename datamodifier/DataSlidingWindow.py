#import the pandas library
import pandas as pd
#import the raw dataset
dataset = pd.read_csv("./originalData/AAPL.csv")
#the new converted sataset containing averages of all categories for specified time frame
newDataSet = pd.DataFrame()
#the y true data set created using the trend categories: increasing, decreasing or consolidating
oneHotDataSet = pd.DataFrame()

#data average, normalization, and ytrue creation class
class SlidingAverage:

    #the total range for the sliding window to parse by
    def __init__(self, k):
        self.k = k

    #create the new data set usng the average of the data within the sliding range
    def average(self, header):
        #initialize the queue for k values
        queue = []
        newDataSet[header] = ""
        #checks k is in the bounds of the dataset
        if(self.k - 1 < dataset[header].shape[0]):
            #initializes the queue with the first k data points
            for i in range(self.k):
                queue.append(dataset[header].at[i])
        #initializes the first pointer to zero
        self.p1 = 0
        #loops through all values of the provided column
        while(True):
            #averages the queue
            queueAverage = sum(queue) / len(queue)
            #adds the obtained average to the new dataset
            newDataSet.loc[self.p1,header] = queueAverage
            #increases the first pointer by one
            self.p1 += 1
            #sets the second pointer to pointer 1 index offset by k
            self.p2 = self.p1 + self.k
            #Compares the second pointer to the shape of the rows and breaks if the same
            if self.p2 == dataset.shape[0]:
                print(f"Pointer 2 is size {self.p2}")
                break
            #removes the first element of the queue
            queue.pop()
            #adds a new value at the second pointer
            queue.append(dataset[header].at[self.p2])
        #displays new dataset to be observed for error
        print(newDataSet)

    def direction(self):
        #initialize k as 1 as you are only comparing the first and second term
        self.k = 1
        self.p1 = 0
        #initializes dataframe columns if the ytrue set is empty
        if len(oneHotDataSet) == 0:
            oneHotDataSet["Increasing"] = ""
            oneHotDataSet["Decreasing"] = ""
            oneHotDataSet["Consolidating"] = ""
        #loops through all data points checking for the conditions of increase, decrease and consolidation
        #where increase is higher highs and higher lows, decrease is lower highs and lower lows and consolidation
        #is anything else
        while(True):
            self.p2 = self.p1 + self.k
            if self.p2 == newDataSet.shape[0]:
                print(f"Pointer 2 is size {self.p2}")
                break
            if (newDataSet["High"].at[self.p1] > newDataSet["High"].at[self.p2]) and (newDataSet["Low"].at[self.p1] > newDataSet["Low"].at[self.p2]):
                oneHotDataSet.loc[self.p1, "Increasing"] = 0
                oneHotDataSet.loc[self.p1, "Decreasing"] = 1
                oneHotDataSet.loc[self.p1, "Consolidating"] = 0
            elif (newDataSet["High"].at[self.p1] < newDataSet["High"].at[self.p2]) and (newDataSet["Low"].at[self.p1] < newDataSet["Low"].at[self.p2]):
                oneHotDataSet.loc[self.p1, "Increasing"] = 1
                oneHotDataSet.loc[self.p1, "Decreasing"] = 0
                oneHotDataSet.loc[self.p1, "Consolidating"] = 0
            else:
                oneHotDataSet.loc[self.p1, "Increasing"] = 0
                oneHotDataSet.loc[self.p1, "Decreasing"] = 0
                oneHotDataSet.loc[self.p1, "Consolidating"] = 1
            #increase the first pointer
            self.p1 += 1
        #print the ytrue data set to be observed for error
        print(oneHotDataSet)

    #normalizes data to prevent overflow error is achieved column by column parsing and averaging
    #the standard deviation is then calculated and the distance of each datapoint from the standard deviation
    #then adds the distance to the normalized mean to get the normalized value
    def normalize(self):
        self.p1 = 0
        headers = list(newDataSet)
        for i in range(len(headers)):
            self.p1 = 0
            header = headers[i]
            print(header)
            mean = newDataSet[header].mean()
            std = newDataSet[header].std()
            flag = True
            while(flag):
                if(self.p1 == newDataSet.shape[0]):
                    print(f"Pointer position {self.p1}")
                    flag = False
                else:
                    currentVal = newDataSet[header].at[self.p1]
                    if(currentVal == mean):
                        newDataSet.loc[self.p1, header] = 3
                    else:
                        distance = (currentVal - mean) / std
                        newDataSet.loc[self.p1, header] = 3 + distance
                    self.p1 += 1
        print(newDataSet)
        
    #converts the ytrue and new data sets to csvs to be used in the model
    def output(self):
        newDataSet.to_csv("./modData/X.csv", index = False)
        oneHotDataSet.to_csv("./modData/Y.csv", index = False)
            
                


#set the date range as k param
getAverage = SlidingAverage(5)
#add the average of defined headers making sure to include "High" and "Low"
getAverage.average("Open")
getAverage.average("Close")
getAverage.average("High")
getAverage.average("Low")
#create one hot vectors
getAverage.direction()
#normalize the averaged dataset
getAverage.normalize()
#create the X.csv and y.csv files for the network
getAverage.output()
            


