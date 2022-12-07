import pandas as pd

def main():
    print('Welcome to Rayyaan\'s Feature Selection Algorithm.')
    fname = input('Type in the name of the file to test : ')
    algoChoice = input('Type in the number of the algorithm you want to run.\n    1) Forward Selection\n    2) Backward Elimination\n')
    dataset = loadFile(fname)
    



def forwardSelection(dataset):
    print('Forward Selection')

def backwardElimination(dataset):
    print('Backward Elimination')

def nearestNeighbour(fname):
    print('Nearest Neighbour')

#function to load /Users/rayyaan/Documents/GitHub/Feature-Selection-with-Nearest-Neighbor/CS170_Small_Data__64.txt
def loadFile(fname):
    dataset = pd.read_fwf(fname, header=None)
    return dataset


    
if __name__ == '__main__':
    main()