import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    # : train Decision Tree
    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        num_cls = np.unique(labels)
        available_attribute_indexes = []
        for index in range(0, len(features[0])):
            available_attribute_indexes.append(index)
        self.root_node = TreeNode(features, labels, num_cls, None, available_attribute_indexes)
        self.root_node.split()

    # predic function
    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        predictions = []
        for feature in features:
            predictions.append(self.root_node.predict(feature))

class TreeNode(object):
    def __init__(self, features, labels, num_cls, parent, available_attribute_indexes):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.parent = parent
        self.available_attribute_indexes
        self.prediction_label = None

        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #  implement split function
    def split(self):
        # if we have no more examples, return majority label of parent.
        if len(labels) == 0:
            self.prediction_label = self.parent.cls_max
            return
        # if we have no more features, return the majority label
        if len(available_feature_indexes) == 0:
            self.prediction_label = self.cls_max
            return

        # if not splittable, return majority label
        if not self.splittable:
            self.prediction_label = self.cls_max
            return

        unique, counts = np.unique(self.labels, return_counts=True)
        current_entropy = Utils.get_entropy(counts) 

        max_info_gain = None
        # find attribute to split on using Info Gain.
        #for index in range(0, len(self.features[0])): 
        for index in self.available_attribute_indexes: 
            branches = self.create_branches(index)
            info_gain = Utils.Information_Gain(current_entropy, branches)
            if max_info_gain is None or info_gain > max_info_gain:
                max_info_gain = info_gain
                self.dim_split = index
            elif info_gain == max_info_gain:
                current_index_values = self.get_num_possible_attribute_values(index)
                best_index_values = self.get_num_possible_attribute_values(self.dim_split)
                if current_index_values > best_index_values:
                    self.dim_split = index

        # remove the feature we are splitting on from the available indexes.
        self.available_attribute_indexes.remove(self.dim_split)

        # Split into child nodes
        attribute_val_dict = {}
        for index in range(0, len(self.features)):
            attribute_val = self.features[index][self.dim_split]
            # first index will be features, second index will be label
            features_and_labels = [[],[]]
            if attribute_val in attribute_val_dict:
                features_and_labels = attribute_val_dict[attribute_val] 
            features_and_labels[0].append(self.features[index])
            features_and_labels[1].append(self.labels[index])

            attribute_val_dict[attribute_val] = features_and_labels

        self.feature_uniq_split = list(attribute_val_dict.keys())
        self.attribute_val_to_child_node = {}
        for key in attribute_val_dict:
            features_and_labels = attribute_val_dict[key]
            child_node = TreeNode(
                    features_and_labels[0],
                    features_and_labels[1],
                    self.num_cls,
                    self,
                    self.available_feature_indexes)  
            if child_node.splittable:
                child_node.split()
            self.children.append(child_node)
            self.attribute_val_to_child_node[key] = child_node

    def get_num_possible_attribute_values(index):
        values = set()
        for feature in self.features:
            values.add(feature[index])
        return len(set)

    def create_branches(self, attribute_index):
        attribute_to_classes = {}
        for feature_index in range(0, len(self.features[0])):
            attribute_val = self.features[feature_index][attribute_index]

            classes = []
            if attribute_val in attribute_to_classes:
                classes = attribute_to_classes[attribute_val]
            classes.append(self.labels[feature_index])
            attribute_to_classes[attribute_val] = classes

        branches = []
        for key in attribute_to_classes:
            classes = attribute_to_classes[key]
            unique, counts = np.unique(classes, return_counts=True)
            branches.append(counts)

        return branches

    # treeNode predict function
    def predict(self, feature):
        if self.prediction_label is not None:
            return self.prediction_label

        attribute_val = feature[self.dim_split]
        child_node = self.attribute_val_to_child_node[attribute_val]
        return child_node.predict()