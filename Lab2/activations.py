import numpy as np

#softmax function turns a vector x of k values into a vector of k real values that sum to 1
#transforms all the values in real values between 0 and 1, so that they can be interpreted as probabilities

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    # TODO your code here
    
    #x - input vector
    #e -  exponential function
    #sum - normalization term 
    
    temperateX = x / t
    e = np.exp(temperateX - np.max(temperateX))
    sum = np.sum(e)
    return e / sum

    # end TODO your code here