import numpy as np
import add_noise as adn
from scipy import ndimage
from tqdm import tqdm

def l1_norm(patch1, patch2):
	# L1 norm difference to compare to different patches
	# Need to apply median filter before computing L1 norm difference
	p1 = np.zeros(np.shape(patch1))
	p2 = np.zeros(np.shape(patch2)) 
	p1[:,:,0], p1[:,:,1], p1[:,:,2] = ndimage.median_filter(patch1[:,:,0],5), ndimage.median_filter(patch1[:,:,1],3), ndimage.median_filter(patch1[:,:,2],5)
	p2[:,:,0], p2[:,:,1], p2[:,:,2] = ndimage.median_filter(patch2[:,:,0],5), ndimage.median_filter(patch2[:,:,1],3), ndimage.median_filter(patch2[:,:,2],5)

	x1 = p1.flatten()
	x2 = p2.flatten()
	return np.sum(np.abs(x1 - x2))


def exhaustive_block_search(video, ref_patch, t, x, y, S = 2, W = 4):
	# video: video vector to perform search in
	# ref_patch: self explanatory

	# t: frame index to perform search in
	# x, y: center of search window
	# S: step size for search window
	# 2*W: patch window size


	# Format of metadata
	# [frame_in_t, x_coordinate, y_coordinate, l1_norm of difference]
	
	patches_metadata = [] # list of patch metadata with patch closest to ref. patch in frame t  
	patch_diff_max = np.inf # keeping track of patch with the max. diff with the reference patch among the current top 5

	L, M, N = np.shape(video)[0], np.shape(video)[1], np.shape(video)[2]
	
	# Define locations for search region
	m_1, m_2 = max(x - S - W, 0) + W, min(x + S + W, M) - W 
	n_1, n_2 = max(y - S - W, 0) + W, min(y + S + W, N) - W

	for m in range(m_1, m_2, S):
		for n in range(n_1, n_2, S):
			patch = video[t, m-W:m+W, n-W:n+W]
			patch_diff = l1_norm(ref_patch, patch)

			# If the diff is less than the maximum of top 5, add it to patches_metadata 
			if patch_diff < patch_diff_max or len(patches_metadata) <= 5:
				patches_metadata.append([t,m,n,patch_diff])
				# Also update the maximum diff value of patches_metadata 
				patch_diff_idx = -1
				patch_diff_max = -np.inf
				for idx in range(len(patches_metadata)):
					patch_metadata = patches_metadata[idx]
					if patch_diff_max < patch_metadata[3]:
						patch_diff_max = patch_metadata[3]
						patch_diff_idx = idx
				if len(patches_metadata) > 5:
					patches_metadata.pop(patch_diff_idx)

	return patches_metadata


def TSS_search(video, ref_patch, t, x, y, S = 4, W = 4):
	patches_metadata = []
	patch_diff_max = np.inf

	L, M, N = np.shape(video)[0], np.shape(video)[1], np.shape(video)[2]
	m_1, m_2 = max(x - S - W, 0) + W, min(x + S + W, M) - W 
	n_1, n_2 = max(y - S - W, 0) + W, min(y + S + W, N) - W

	# Search in step sizes of +/- 4 and keep track of the minimum 
	for step in [S, S/2, S/4]:
		for m in range(m_1, m_2, step):
			for n in range(n_1, n_2, step):
				patch = video[t, m-W:m+W, n-W:n+W]
				patch_diff = l1_norm(ref_patch, patch)

				# If the diff is less than the maximum of top 5, add it to patches_metadata 
				if patch_diff < patch_diff_max or len(patches_metadata) <= 5:
					patches_metadata.append([t,m,n,patch_diff])
					# Also update the maximum diff value of patches_metadata 
					patch_diff_idx = -1
					max_diff_max = -np.inf
					patch_diff_min = np.inf
					min_diff_idx = 0

					for idx in range(len(patches_metadata)):
						patch_metadata = patches_metadata[idx]
						if patch_diff_max < patch_metadata[3]:
							patch_diff_max = patch_metadata[3]
							max_diff_idx = idx
						if patch_diff_min > patch_metadata[3]:
							patch_diff_min = patch_metadata[3]
							min_diff_idx = idx
					if len(patches_metadata) > 5:
						patches_metadata.pop(patch_diff_idx)

		x, y = patches_metadata[min_diff_idx][1], patches_metadata[min_diff_idx][2]
		m_1, m_2 = max(x - step/2 - W, 0) + W, min(x + step/2 + W, M) - W 
		n_1, n_2 = max(y - step/2 - W, 0) + W, min(y + step/2 + W, N) - W
	return patches_metadata

def denoise_video_frame(video, t, S = 4, W = 4, search_method = 'exhuastive', verbose = False):
	# S: step size for ref. patch movement
	L, M, N = np.shape(video)[0], np.shape(video)[1], np.shape(video)[2]
	frame_denoised = np.zeros((L,M,N,3))
	frame_denoised_count = (0.00001)*np.ones((L,M,N,3))


	L, M, N = np.shape(video)[0], np.shape(video)[1], np.shape(video)[2]
	# Search for patches in relevant frames
	for m in tqdm(range(S,M-S,S)):
		for n in range(S,N-S,S):
			list_patches_metadata = []
			ref_patch = video[t, m-W:m+W, n-W:n+W]	
			for l in range(L):
				if search_method == 'exhuastive':
					patches_metadata = exhaustive_block_search(video, ref_patch, l, m, n, S, W)
				if search_method == 'TSS':
					patches_metadata = TSS_search(video, ref_patch, l, m, n, S, W)
				list_patches_metadata.append(patches_metadata)
			# P_matrx constructed here.
			P_matrx = construct_P(video, list_patches_metadata, W)
			# Denoising P_matrx: Q_matrx
			Q_matrx = P_matrx
			# Q_matrx is used for patch grouping instead of P_matrx
			frame_denoised, frame_denoised_count = reconstruct_frames(Q_matrx, list_patches_metadata, frame_denoised, frame_denoised_count, W)
	return np.divide(frame_denoised, frame_denoised_count)

def construct_P(video, list_patches_metadata, W):
	P_matrx = []
	for j in range(len(list_patches_metadata)):
		patches_metadata = list_patches_metadata[j]
		for i in range(len(patches_metadata)):
			patch_t, patch_x, patch_y = patches_metadata[i][0], patches_metadata[i][1], patches_metadata[i][2]
			patch = video[patch_t, patch_x-W:patch_x+W, patch_y-W:patch_y+W]
			P_matrx.append(patch.flatten())	
	return P_matrx.asarray(P_matrx, dtype = np.float32)

def reconstruct_frames(P_matrx, list_patches_metadata, frame_denoised, frame_denoised_count, W = 4):
	
	count = 0;
	for i in range(len(list_patches_metadata)):
		patches_metadata = list_patches_metadata[i]
		for j in range(len(patches_metadata)):
			t, x, y = patches_metadata[j][0], patches_metadata[j][1], patches_metadata[j][2]
			frame_denoised[t, x-W:x+W, y-W:y+W] = frame_denoised[t, x-W:x+W, y-W:y+W]+ P_matrx[count,:].reshape(2*W,2*W,3)
			frame_denoised_count[t, x-W:x+W, y-W:y+W] = frame_denoised_count[t, x-W:x+W, y-W:y+W] + 1
			count += 1
	return [frame_denoised, frame_denoised_count]
