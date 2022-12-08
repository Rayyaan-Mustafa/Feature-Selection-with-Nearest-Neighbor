import numpy as np
import math
import time
import copy

def main():
    print('Welcome to Rayyaan\'s Feature Selection Algorithm.')
    fname = input('Type in the name of the file to test : ')
    algoChoice = input('Type in the number of the algorithm you want to run.\n    1) Forward Selection\n    2) Backward Elimination\n')
    data = np.loadtxt(fname) #read the file into an numpy array
    print("This datast has " + str(len(data[0]) - 1) + " features (not including the class attribute), with " + str(len(data)) + " instances.")
    print("Running nearest neighbor with all " + str(len(data[0]) - 1) + " features, using \"leaving-one-out\" evaluation, I get an accuracy of " + str(leave_one_out_cross_validation(data, [], 0)*100) + "%")
    print("Beginning search")
    start = time.time()
    end = 0
    #forward selection
    if algoChoice == '1':
        overall_best_accuracy = 0
        overall_best_set = []
        current_set_of_features = [] #initialize empty set
        for i in range(1, len(data[0])):
            # print("On the " + str(i) + "th level of the search tree")
            feature_to_add_at_this_level = None
            best_accuracy_so_far = 0
            for k in range(1,len(data[0])):
                if k not in current_set_of_features:#only consider adding, if not already added
                    # print("--Considering adding the " + str(k) + " feature")
                    accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                    if accuracy > best_accuracy_so_far:
                        best_accuracy_so_far = accuracy
                        feature_to_add_at_this_level = k
                        if best_accuracy_so_far > overall_best_accuracy:
                            overall_best_accuracy = best_accuracy_so_far
                            current_set_of_features.append(feature_to_add_at_this_level)
                            overall_best_set = current_set_of_features
            # print("On level " + str(i) + " I added feature " + str(feature_to_add_at_this_level) + " to the current set")
            print("Feature set " + str(current_set_of_features) + " was best, accuracy is " + str(best_accuracy_so_far*100) + "%")
        print("Finished search!! The best feature subset is " + str(overall_best_set) + ", which has an accuracy of " + str(overall_best_accuracy*100) + "%")
        end = time.time()
    #backward elimination
    elif algoChoice == '2':
        #insert backward elimination code here
        end = time.time()
    else:
        print("Incorrect algoritm choice. Please try again.")
        return
    


def leave_one_out_cross_validation(data, current_set_of_features, feature_to_add):
    # return np.random.rand()
    current_set_of_features_copy = copy.deepcopy(current_set_of_features)
    if feature_to_add != 0:
        current_set_of_features_copy.append(feature_to_add)
    number_correctly_classified = 0
    data_copy = copy.deepcopy(data)
    unused_features = set(range(1,len(data[0]))) - set(current_set_of_features_copy)
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
                distance = math.sqrt(sum((object_to_classify - data_copy[k][1:])**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data_copy[nearest_neighbor_location][0]
        # print("Object " + str(i) + " is class " + str(label_object_to_classify))
        # print("It's nearest neighbor is " + str(nearest_neighbor_location) + " which is in class " + str(nearest_neighbor_label))
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / len(data_copy)
    # print("Accuracy is " + str(accuracy))
    print("\tUsing feature(s) " + str(current_set_of_features_copy) + " accuracy is " + str(accuracy*100) + "%")
    return accuracy

if __name__ == '__main__':
    main()