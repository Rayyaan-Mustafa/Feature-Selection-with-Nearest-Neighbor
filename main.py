import numpy as np
import math
import time
import copy
import logging

#tests to get the numbers for multiple files at once
def automate_tests():
    files = ["CS170_Small_Data__64.txt","CS170_Large_Data__99.txt"]
    try:
        logging.info("Starting tests")
        logging.info("Testing forward selection")
        for f in files:
            data = np.loadtxt(f) #read the file into an numpy array
            start = time.time()
            logging.info("Testing file: " + f)
            forward_selection(data)
            end = time.time()
            logging.info("Time taken: " + str(round(end-start,2)) + " seconds")
        logging.info("Testing backward elimination")
        for f in files:
            data = np.loadtxt(f) #read the file into an numpy array
            start = time.time()
            logging.info("Testing file: " + f)
            backward_elimination(data)
            end = time.time()
            logging.info("Time taken: " + str(round(end-start,2)) + " seconds")
    except Exception as e:
        logging.error("Error in testing: " + str(e))
    
def main():
    print('Welcome to Rayyaan\'s Feature Selection Algorithm.')
    fname = input('Type in the name of the file to test : ')
    algoChoice = input('Type in the number of the algorithm you want to run.\n    1) Forward Selection\n    2) Backward Elimination\n')
    data = np.loadtxt(fname) #read the file into an numpy array
    print("This dataset has " + str(len(data[0]) - 1) + " features (not including the class attribute), with " + str(len(data)) + " instances.")
    print("Running nearest neighbor with all " + str(len(data[0]) - 1) + " features, using \"leaving-one-out\" evaluation, I get an accuracy of " + str(round(leave_one_out_cross_validation(data, set(range(1,len(data[0]))), 0,0)*100,2)) + "%")
    print("Beginning search")
    start = time.time()
    #forward selection
    if algoChoice == '1':
        forward_selection(data)
    elif algoChoice == '2':
        backward_elimination(data)
    else:
        print("Incorrect algoritm choice. Please try again.")
        return
    end = time.time()
    print("Search took " + str(round(end-start,1)) + " seconds.")#spurious precision!!
    

def forward_selection(data):
    OVERALL_best_accuracy = 0
    OVERALL_best_set = []
    current_set_of_features = [] #initialize empty set
    for i in range(1, len(data[0])):#iterate through all features
        # print("On the " + str(i) + "th level of the search tree")
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0
        for k in range(1,len(data[0])):#iterate through all features
            if k not in current_set_of_features:#only consider adding, if not already added
                # print("--Considering adding the " + str(k) + " feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k,1)#calculate accuracy when you add feature k
                if accuracy > best_accuracy_so_far:#if accuracy is better than best so far, update best accuracy and feature to add
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        if best_accuracy_so_far > OVERALL_best_accuracy:#if best_accuracy_so_far is better than OVERALL_best_accuracy, update OVERALL_best_accuracy and OVERALL_best_set
            OVERALL_best_accuracy = best_accuracy_so_far
            current_set_of_features.append(feature_to_add_at_this_level)
            OVERALL_best_set = copy.deepcopy(current_set_of_features)
        else:
            current_set_of_features.append(feature_to_add_at_this_level)
        # print("On level " + str(i) + " I added feature " + str(feature_to_add_at_this_level) + " to the current set")
        print("Feature set " + str(current_set_of_features) + " was best, accuracy is " + str(round(best_accuracy_so_far*100,2)) + "%")
    print("Finished search!! The best feature subset is " + str(OVERALL_best_set) + ", which has an accuracy of " + str(OVERALL_best_accuracy*100) + "%")
    logging.info("Finished search!! The best feature subset is " + str(OVERALL_best_set) + ", which has an accuracy of " + str(OVERALL_best_accuracy*100) + "%")
    
def backward_elimination(data):
    OVERALL_best_accuracy = 0
    OVERALL_best_set = []
    current_set_of_features = set(range(1,len(data[0])))#initialize set to all features
    for i in range(1, len(data[0])):
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0
        for k in range(1,len(data[0])):
            if k in current_set_of_features:#only consider removing, if not already removed
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k,2)#calculate accuracy when you remove feature k
                if accuracy > best_accuracy_so_far:#if accuracy is better than best so far, update best accuracy and feature to remove
                    best_accuracy_so_far = accuracy
                    feature_to_remove_at_this_level = k
        if best_accuracy_so_far > OVERALL_best_accuracy:#if best_accuracy_so_far is better than OVERALL_best_accuracy, update OVERALL_best_accuracy and OVERALL_best_set
            OVERALL_best_accuracy = best_accuracy_so_far
            current_set_of_features.remove(feature_to_remove_at_this_level)
            OVERALL_best_set = copy.deepcopy(current_set_of_features)
        else:
            current_set_of_features.remove(feature_to_remove_at_this_level)
        print("Feature set " + str(current_set_of_features) + " was best, accuracy is " + str(best_accuracy_so_far*100) + "%")
    print("Finished search!! The best feature subset is " + str(OVERALL_best_set) + ", which has an accuracy of " + str(OVERALL_best_accuracy*100) + "%")
    logging.info("Finished search!! The best feature subset is " + str(OVERALL_best_set) + ", which has an accuracy of " + str(OVERALL_best_accuracy*100) + "%")

    

def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add_or_remove, algo):
    # return np.random.rand()
    current_set_of_features_copy = copy.deepcopy(current_set_of_features)#make a copy of the current set of features to avoid modifying the original
    if algo == 1:#if we are adding a feature (forward selection)
        if feature_to_add_or_remove != 0:
            current_set_of_features_copy.append(feature_to_add_or_remove)
    elif algo == 2:#if we are removing a feature (backward elimination)
        if feature_to_add_or_remove != 0:
            current_set_of_features_copy.remove(feature_to_add_or_remove)
    number_correctly_classified = 0
    data_copy = copy.deepcopy(data)#make a copy of the data to avoid modifying the original
    unused_features = set(range(1,len(data[0]))) - set(current_set_of_features_copy)
    # print("current set of features: " + str(current_set_of_features_copy))  
    # print("feature_to_add_or_remove" + str(feature_to_add_or_remove))
    for i in range(0, len(data_copy)):
        for f in unused_features:
                data_copy[i][f] = 0 #(Slide 65) set the feature values of the unused features to 0
    for i in range(0, len(data_copy)):#for each row 'i' in the data_copy
        object_to_classify = data_copy[i][1:]#selecting all the features of a row
        label_object_to_classify = data_copy[i][0]#selcting the label of a row(1 or 2)

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(0,len(data_copy)):#for each row 'k' in the data_copy
            if k != i:
                distance = math.sqrt(sum((object_to_classify - data_copy[k][1:])**2))#calculating distance between object_to_classify and row 'k'
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data_copy[nearest_neighbor_location][0]
        # print("Object " + str(i) + " is class " + str(label_object_to_classify))
        # print("It's nearest neighbor is " + str(nearest_neighbor_location) + " which is in class " + str(nearest_neighbor_label))
        if label_object_to_classify == nearest_neighbor_label:#
            number_correctly_classified += 1
    accuracy = number_correctly_classified / len(data_copy)
    # print("Accuracy is " + str(accuracy))
    print("\tUsing feature(s) " + str(current_set_of_features_copy) + " accuracy is " + str(round(accuracy*100,2)) + "%")
    return accuracy

if __name__ == '__main__':

    main()
    # logging.basicConfig(filename='app.log' + str(time.time()), filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
    # automate_tests()