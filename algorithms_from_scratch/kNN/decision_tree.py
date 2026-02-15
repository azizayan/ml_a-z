import numpy as np

def calculate_impurity(labels):

    values, counts = np.unique(labels,return_counts= True)

    probabilities = []

    for i  in range(len(values)):
        value_prob = counts[i] / len(labels)

        probabilities.append(value_prob)

    squares = 0
    for i in range(len(probabilities)):
        
        squares += probabilities[i] * probabilities[i]

    return 1 - squares


def best_split(features, labels):
    
    best_gain = -1
    split_index = None
    split_threshold = None


    n_samples, n_features = features.shape

    parent_impurity = calculate_impurity (labels)

    for feature_index in range(n_features):
        
        feature_column = features[:, feature_index]

        thresholds = np.unique(feature_column)


        for threshold in thresholds:

            left_mask = feature_column <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue


            y_left = y[left_mask]
            y_right = y[right_mask]

            imp_left = calculate_impurity(y_left)
            imp_right = calculate_impurity(y_right)
            
           
            n_l = len(y_left)
            n_r = len(y_right)
            weight_l = n_l / n_samples
            weight_r = n_r / n_samples
            
            
            child_impurity = (weight_l * imp_left) + (weight_r * imp_right)
            
            
            gain = parent_impurity - child_impurity
            
           
            if gain > best_gain:
                best_gain = gain
                best_index = feature_index
                best_threshold = threshold

    
    return best_index, best_threshold





