import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # compute the loss
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    # ensure that the exp won't be too big
    scores = X[i].dot(W)
    shift_scores = scores - np.max(scores)
    p = np.exp(shift_scores)/np.sum(np.exp(shift_scores))
    loss -= np.log(p[y[i]])
    # compute dW
    for j in xrange(num_classes):
      softmax_derive = np.exp(shift_scores[j])/np.sum(np.exp(shift_scores))
      if j == y[i]:
        # using the vector chain rule in matrix calculus to get the result
        # note X dot W as f, and Li is a function of fi
        # then calculate the partial derivitive of fi over W
        dW[:,j] += (softmax_derive-1)*X[i]
      else:
        dW[:,j] += softmax_derive*X[i]
  
  loss /= num_train
  dW = dW/num_train + reg*W
  loss += reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  # compute the loss
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
  # broadcast (500, 10) with (500, 1)
  p = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(num_train, 1) # (500, 10)
  loss -=np.sum(np.log(p[np.arange(num_train), y]))
  loss /= num_train
  # add the regularization to the loss
  loss += reg*np.sum(W*W)
  # compute the derivative
  p[np.arange(num_train), y] -= 1
  # X.T dot p is the last part of the chain
  dW = np.dot(X.T, p)
  dW = dW/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

