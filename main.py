import pandas as pd
import math

def main():
    print('Welcome to Rayyaan\'s Feature Selection Algorithm.')
    fname = input('Type in the name of the file to test : ')
    algoChoice = input('Type in the number of the algorithm you want to run.\n    1) Forward Selection\n    2) Backward Elimination\n')
    data = loadFile(fname)
    

    number_correctly_classified = 0
    for i in range(0, len(data)):#for each row 'i' in the data
        object_to_classify = data.iloc[i][1:]#selecting all the features of a row
        label_object_to_classify = data.iloc[i][0]#selcting the label of a row(1 or 2)

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(0,len(data)):#for each row 'k' in the data
            if k != i:
                distance = math.sqrt(sum(object_to_classify - data.iloc[k][1:])**2)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data.iloc[nearest_neighbor_location][0]
        print("Object " + str(i) + " is class " + str(label_object_to_classify))
        print("It's nearest neighbor is " + str(nearest_neighbor_location) + " which is in class " + str(nearest_neighbor_label))
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
        accuracy = number_correctly_classified / len(data)
        print("Accuracy is " + str(accuracy))
    if algoChoice == '1':
        forwardSelection(data)
    elif algoChoice == '2':
        backwardElimination(data)
    else:
        print("Incorrect algoritm choice. Please try again.")



def forwardSelection(data):
    print('Forward Selection')

def backwardElimination(data):
    print('Backward Elimination')

def nearestNeighbour(fname):
    print('Nearest Neighbour')

#function to load /Users/rayyaan/Documents/GitHub/Feature-Selection-with-Nearest-Neighbor/CS170_Small_Data__64.txt
def loadFile(fname):
    data = pd.read_fwf(fname, header=None) #read the file into a pandas dataframe and get rid of header
    return data


    
if __name__ == '__main__':
    main()