def get_omega_mask(noisy_mat, std_dev_mul):

	# returns
	# 	omega - mask denoting which elements to throw away

	# parameters
	# 	noisy_mat - the matrix obtained after patch matching and 
	# 	median filtering
	
	# 	std_dev_mul - the factor with which to multiply the standard
	# 	deviation to find the elements to throw away. For example, if 
	# 	this parameter is set to 2, all elements of each row lying between
	# 	mean-2*std_deviation to mean+2*std_deviation will be thrown away
	# 	- which means the mask elements corresponding to these elements 
	# 	will be set to 0

	omega = np.array()

	for row in noisy_mat:

		m = np.mean(row)
		s = np.std(row)

		row_mask = np.zeros(row.shape)

		for i,elem in enumerate(row):

			if(elem>=m-std_dev_mul*s and elem<=m+std_dev_mul*s):
				row_mask[i] = 1

		omega = np.stack((omega, row_mask))


	return omega



