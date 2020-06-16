import pandas as pd
import numpy as np
from collections import Counter



class ID3():
    
    def __init__(self,col_names):
        '''
        col_names: column names of attributes
        '''
        self.col_names = col_names
    
    
    def transform_structure(self,X,y=None):
        '''
        This will change the data type from X,y arrays to a list of dictonaries format
        '''
        lis_dic = []
        for i in range(len(X)):
            dic = {}
            for j in range(len(X[i])):
                dic[self.col_names[j]] = X[i][j]
            if y is not None:
                lis_dic.append((dic,y[i]))
            else:
                lis_dic.append(dic)
        return lis_dic


    def compute_entropy(self,class_probabilities):
        '''
        class_probabilities is a list of class probabilities
        '''
        terms = [-pi * np.log2(pi) for pi in class_probabilities if pi] # ignore zero probabilities
        H = np.sum(terms)
        return H


    def compute_class_probabilities(self,instance_labels):
        '''
        instance_labels is a list of each examples' class label
        '''
        num_examples = len(instance_labels)
        counts = list(Counter(instance_labels).values())
        probabilities = np.array(counts) / num_examples
        return probabilities


    def compute_subset_entropy(self,subset):
        '''
        subset is a list of instances as two-item tuples (attributes, label)
        '''
        labels = [label for _, label in subset]
        probabilities = self.compute_class_probabilities(labels)
        entropy = self.compute_entropy(probabilities)
        return entropy


    def compute_partition_entropy(self,subsets):
        '''
        subsets is a list of class label lists
        '''
        num_examples = np.sum([len(s) for s in subsets])
        entropies = [(len(s) / num_examples) * self.compute_subset_entropy(s) for s in subsets]
        partition_entropy = np.sum(entropies)
        return partition_entropy


    def partition_by(self,inputs, attribute):
        '''
        inputs is a list of tuple pairs: (attribute_dict, label)
        attribute is the proposed attribute to partition by
        returns a dictionary of attribute value: input subsets pairs
        '''
        subsets = {}
        for example in inputs:
            attribute_value = example[0][attribute]
            if attribute_value in subsets:
                subsets[attribute_value].append(example)
            else: # add this attribute_value to the dict
                subsets[attribute_value] = [example]
        return subsets

    
    def partition_entropy_by(self,inputs, attribute):
        '''
        compute the partition
        compute the entropy of the partition
        '''
        subsets = self.partition_by(inputs, attribute)
        entropies = self.compute_partition_entropy(subsets.values())
        return entropies


    def find_min_entropy_partition(self,inputs, attributes=None):
        '''

        '''
        if attributes is None:
            attributes = list(inputs[0][0].keys())
        partition_entropies = []
        for attribute in attributes:
            partition_entropy = self.partition_entropy_by(inputs, attribute)
            # print(attribute, partition_entropy)
            partition_entropies.append(partition_entropy)
        min_index = np.argmin(partition_entropies)
        return attributes[min_index]

    
    def build_tree(self,inputs, split_candidates=None):
        '''
        implements the ID3 algorithm to build a decision tree
        '''
        if split_candidates is None:
            # this is the first pass
            split_candidates = list(inputs[0][0].keys())

        num_examples = len(inputs)
        # count Trues and Falses in the examples
        num_trues = len([label for attributes, label in inputs if label == True])
        num_falses = num_examples - num_trues

        # part (1) in the ID3 algorithm -> all same class label
        if num_trues == 0: # no trues, this is a False leaf node
            return False
        if num_falses == 0: # no falses, this is a True leaf node
            return True

        # part (2) in the ID3 algorithm -> list of attributes is empty -> leaf node with majority class label
        if not split_candidates: 
            return num_trues >= num_falses

        # part (3) in ID3 algorithm -> split on best attribute
        split_attribute = self.find_min_entropy_partition(inputs, split_candidates)
        partitions = self.partition_by(inputs, split_attribute)
        new_split_candidates = split_candidates[:]
        new_split_candidates.remove(split_attribute)

        # recursively build the subtrees
        subtrees = {}
        for attribute_value, subset in partitions.items():
            subtrees[attribute_value] = self.build_tree(subset, new_split_candidates)

        # missing (or unexpected) attribute value
        subtrees[None] = num_trues > num_falses

        return (split_attribute, subtrees)

    
    def fit(self,X,y):
        self.tree = self.build_tree(self.transform_structure(X,y))
        
        
    def classify(self,tree,new_example):
        '''
        classify new_example using decision tree
        '''
        # leaf node, return value
        if tree in [True, False]:
            return tree

        # decision node, unpack the attribute to split on and subtrees
        attribute, subtree_dict = tree

        subtree_key = new_example.get(attribute) # get return None if attribute not in new_example dict
        if subtree_key not in subtree_dict:
            subtree_key = None # use None subtree if no subtree for key

        subtree = subtree_dict[subtree_key]
        label = self.classify(subtree, new_example)
        return label
    
  
    def predict(self,X):
        '''
        Predict on the training instances
        '''
        X = self.transform_structure(X)
        return [(self.classify(self.tree,i)) for i in X]
    
    
    def get_params(self, deep = False):
        return {'col_names':self.col_names}