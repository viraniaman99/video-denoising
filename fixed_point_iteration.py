import numpy as np
import numpy.linalg.norm 
import numpy.linalg.svd
# import copy

# default norm is frobenius in the above imported method

def fixed_point_iteration(num_iter, epsilon, tau, mu, P, omega):

	# iter - max number of iterations

	# epsilon - error bound on difference between two successive 
	# approximations of the complete matrix

	# tau - soft thresholding constant, between 1 and 2

	# mu - parameter that is part of the lagranian formulation
	# of the problem. fixed constant

	# P - original patch matrix 

	# omega - binary matrix same size as patch matrix that denotes
	# which element is deleted (corresponding entry in omega set to 0),
	# and which element is kept

	# returns the estimated low rank matrix Q


	Q = np.zeros(P.shape)
	Q_prev = np.ones(P.shape)
	Q_next = np.zeros(P.shape)

	while (num_iter>0 and norm(Q-Q_prev)>epsilon) :

		R = Q - tau*(np.multiply(omega, Q-P))
		Q_next = soft_threshold(tau*mu, R)

		# Q_prev = copy.deepcopy(Q)
		# Q = copy.deepcopy(Q_next)

		Q_prev = Q
		Q = Q_next
		num_iter = num_iter-1


	return Q

def soft_threshold(tau, R):

	u, s, vh = svd(R, full_matrices=False)
	s = s - tau
	s = np.maximum(s, 0)
	return u*s*vh



