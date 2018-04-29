import cv2
import sys
import numpy as np
import TSS_block_search as patch_search
import add_noise as adn
from PIL import Image


cap = cv2.VideoCapture(sys.argv[1])
video = []
frameCount = 0
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret:
		video.append(frame)
		frameCount += 1
	else:
		break

video = np.asarray(video)
video_float = np.asarray(video, dtype = np.float32)
video_noisy = adn.add_noise(video_float, 20, 1, 0.01)


denoised_frames = patch_search.search_patch(video_noisy[0:10], t = 4, search_method = 'TSS')
img = Image.fromarray(denoised_frames[4].astype(np.uint8), 'RGB')

img.save(sys.argv[2]+'_denoised.png')

img = Image.fromarray(video_noisy[4].astype(np.uint8), 'RGB')
img.save(sys.argv[2]+'.png')	


# add_noise.test_add_noise(sys.argv[1])