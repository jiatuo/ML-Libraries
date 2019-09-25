from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy
from collections import Counter


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
import hw1_utils

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.training_examples = {}
        for index in range(0,len(features)):
            self.training_examples[tuple(features[index])] = labels[index] 
        
    
    def predict(self, features: List[List[float]]) -> List[int]:
        predicted_classes = []
        for index in range(0,len(features)):
            k_nearest = self.get_k_neighbors(features[index])
            predicted_classes.append(self._get_class_from_neighbors(k_nearest))
        return predicted_classes

    def _get_class_from_neighbors(self, k_nearest):
        counter = Counter(k_nearest)
        label, count = counter.most_common()[0]
        return label
        
        
    # returns the labels for the k nearest neighbors
    def get_k_neighbors(self, point):
        k_nearest = []
        distances_to_features = {} 
        for features in self.training_examples.keys():
            features = list(features)
            distance = self.distance_function(point, features)
            feature_list = []
            if distance in distances_to_features:
                feature_list = distances_to_features[distance]
            feature_list.append(features)
            distances_to_features[distance] = feature_list

        distances = list(distances_to_features.keys())
        distances.sort()
        smallest_distances = distances[0:self.k]
        for distance in smallest_distances:
            feature_list = distances_to_features[distance]
            for feature in feature_list:
                if len(k_nearest) == self.k:
                    return k_nearest
                k_nearest.append(self.training_examples[tuple(feature)])
        return k_nearest

    # Gets the f1 score from given knn_classifier and f1_score function.
    def _get_f1_score_from_classifier(knn_classifier, Xval, yval, f1_score):
        predictions = []
        for features in Xval:
            predictions.append(knn_classifier.predict(features))
        score = f1_score(yval, predictions)
        return score
            

        
    # model selection function where you need to find the best k     
    def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        model = None
        best_k = 0
        current_k = 0
        best_f1_score = None

        while current_k < len(ytrain):
            current_k += 1
            for distance_function_key in distance_funcs.keys():
                distance_function = distance_funcs[distance_function_key]

                knn_classifier = KNN(current_k, distance_function)
                knn_classifier.train(Xtrain, ytrain)
                knn_score = _get_f1_score_from_classifier(knn_classifier, Xval, yVal)
                if best_f1_score == None or best_f1_score < knn_score:
                    best_f1_score = knn_score
                    best_k = current_k
                    model = knn_classifier
                

        
        #Dont change any print statement
        print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

        print()
        print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
              'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
        print()

        return best_k, model
    
    # Complete the model selection function where you need to find the best k with transformation
    def model_selection_with_transformation(distance_funcs,scaling_classes, Xtrain, ytrain, f1_score, Xval, yval, Xtest, ytest):
        model = None
        best_f1_score = None
        best_k = None

        for scaling_class_key in scaling_classes.keys():
            scaling_class = scaling_classes[scaling_class_key]
            scaler = scaling_class()
            scaled_Xtrain = scaler(Xtrain)
            scaled_ytrain = scaler(ytrain)
            scaled_Xval = scaler(Xval)
            scaled_yval = scaler(yval)
            scaled_Xtest = scaler(Xtest)
            scaled_ytest = scaler(ytest)

            best_k_with_scaled , best_model_with_scaled = KNN.model_selection_without_transformation(
                    distance_funcs, scaled_Xtrain, scaled_ytrain, f1_score, scaled_Xval, scaled_yval, scaled_Xtest, scaled_ytest)
            best_f1_score_with_scaled = _get_f1_score_from_classifier(best_model_with_scaled, scaled_Xval, scaled_yval, f1_score)
            if best_f1_score == None or best_f1_score_with_scaled > best_f1_score:
                best_f1_score = best_f1_score_with_scaled
                model = best_model_with_scaled
                best_k = best_k_with_scaled


        
        #Dont change any print statement
        print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) + 
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

        print()
        print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
              'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
        print()
        return best_k, model
        
        
    # Do the classification 
    def test_classify(model):
        from data import data_processing
        
        Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
        model.train(Xtrain, ytrain)
        predicted_labels = model.predict(Xtest)




        

if __name__ == '__main__':
    model = KNN(5, hw1_utils.euclidean_distance)
    KNN.test_classify(model)


    print(np.__version__)
    print(scipy.__version__)
