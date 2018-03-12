import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  print(X.shape, W.shape)# (500, 3073) (3073, 10)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # margin here is equivelent to margin[i, j] (num_trian, num_class)
        # dW[a, b] (pixels, num_class) is the sum of X[:, b] if margin[i, j] > 0
        loss += margin
        # only when the margin is greater than 0 can the W affect L
        dW[:, j] += X[i]
        # ?? the following expression will run <j if(margin > 0)> k times!
        # every time hinge is used all involves minus correct_calss_score
        # so each time we use dW[:, y[i]] its effect on the loss is k*para
        dW[:, y[i]] -= X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # ? why do we need to divide the derivative by num_train
  # answer: the loss is divide by num_train which change the parameters of the function
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  score = X.dot(W)
  # get the correct score and make it a matrix
  # currently its a one dimension array
  correct_class_score = score[np.arange(num_train), y]
  # make it a matrix
  correct_class_score = correct_class_score.reshape(num_train, 1)
  correct_class_score = np.repeat(correct_class_score, num_classes, axis = 1)
  # using hinge_function to calculate the margin
  margin = score - correct_class_score + 1
  #print(correct_class_score)
  # the value of correct_class is 0
  margin[np.arange(num_train), y] = 0
  # compute the loss
  loss = (np.sum(margin[margin > 0]))/num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin_sign = np.zeros(margin.shape)
  margin_sign[margin > 0] = 1
  # explanation ahead!!!
  margin_sign[np.arange(num_train), y] = 0
  row_sum = np.sum(margin_sign, axis=1)
  margin_sign[np.arange(num_train), y] = -row_sum
  #print(row_sum)
  dW += np.dot(X.T, margin_sign)
  dW /= num_train
  dW += 2*reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
