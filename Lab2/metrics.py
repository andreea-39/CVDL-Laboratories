import numpy as np



#A confusion matrix is a summary of prediction results on a classification problem.
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes = None) -> np.ndarray:
    """"
    Computes the confusion matrix from labels (y_true) and predictions (y_pred).
    The matrix columns represent the prediction labels and the rows represent the ground truth labels.
    The confusion matrix is always a 2-D array of shape `[num_classes, num_classes]`,
    where `num_classes` is the number of valid labels for a given classification task.
    The arguments y_true and y_pred must have the same shapes in order for this function to work
    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    if num_classes == None:
        num_classes = max(np.argmax(y_true), np.argmax(y_pred))
    conf_mat = np.zeros((num_classes, num_classes))
    # TODO your code here - compute the confusion matrix
    # even here try to use vectorization, so NO for loops

    # 0. if the number of classes is not provided, compute it based on the y_true and y_pred arrays

    # 1. create a confusion matrix of shape (num_classes, num_classes) and initialize it to 0

    # 2. use argmax to get the maximal prediction for each sample
    # hint: you might find np.add.at useful: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html

    np.add.at(conf_mat, (y_true, y_pred), 1)

    # end TODO your code here
    return conf_mat


#TP / (TP + FP)
#Precision is the ratio between the True Positives and all the Positives.
def precision_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None) -> float:
    """"
    Computes the precision score.
    For binary classification, the precision score is defined as the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives.
    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.
    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    precision = 0
    # TODO your code here
    conf_mat = confusion_matrix(y_true, y_pred, num_classes)

    tp = np.diag(conf_mat)
    all_pred = np.sum(conf_mat, axis=0)
    precision = np.average(tp / all_pred)
    # end TODO your code here
    return precision

# TP / (TP + FN)
#The recall is the measure of our model correctly identifying True Positives.
def recall_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None)  -> float:
    """"
    Computes the recall score.
    For binary classification, the recall score is defined as the ratio tp / (tp + fn)
    where tp is the number of true positives and fn the number of false negatives
    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.
    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    recall = None
    # TODO your code here
    conf_mat = confusion_matrix(y_true, y_pred, num_classes)

    tp = np.diag(conf_mat)
    all_pred = np.sum(conf_mat, axis=1)
    recall = np.average(tp / all_pred)

    # end TODO your code here
    return recall

# (TP + TN)/(TP + FP + TN +FN)
#Accuracy is the ratio of the total number of correct predictions and the total number of predictions.
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc_score = 0
    # TODO your code here
    # remember, use vectorization, so no for loops
    # hint: you might find np.trace useful here https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    tp = np.sum(y_true == y_pred)
    all = len(y_true)
    acc_score = tp / all
    # end TODO your code here
    return acc_score


if __name__ == '__main__':
    pass
    # TODO your tests here
    # add some test for your code.
    # you could use the sklean.metrics module (with macro averaging to check your results)