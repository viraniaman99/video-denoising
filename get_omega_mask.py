def get_omega_mask(noisy_mat, std_dev_mul):

	# returns
	# 	omega - mask denoting which elements to throw away

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



