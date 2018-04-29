import cv2
import numpy as np
import sys

def add_noise(video, sigma, k, s):
	# sigma: gaussian noise variance
	# k: poisson noise parameter
	# s: probability of a corrupted pixel
	video_shape = np.shape(video)
	video_noisy = video + np.random.normal(0, sigma, video_shape)
	video_noisy = np.maximum(video_noisy, 0)

	poisson_noise = np.sqrt(k)*(np.random.poisson(video_noisy) - video_noisy)
	video_noisy = video_noisy+poisson_noise

	# Deciding which pxiels to corrupt
	video_un = np.random.normal(0,1,video_shape)
	video_0 =  np.asarray( video_un >  s*0.5, dtype = np.float32)
	video_noisy = np.multiply(video_noisy, video_0)

	video_255 = 255*np.asarray( np.logical_and(video_un > s*0.5, video_un <= s), dtype = np.float32)
	video_noisy = np.minimum(video_255+video_noisy, 255)

	return video_noisy


def PSNR(video, video_noisy):
	err = np.flatten(video-video_noisy)
	N = np.shape(err)[0]
	return 10*np.log10(((255*N)**2)/(np.linalg.norm(err)**2))

def test_add_noise(video_file):
	cap = cv2.VideoCapture(video_file)
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

	cap.release()
	cv2.destroyAllWindows()


	video_noisy = add_noise(video_float, 0, 5, 0.00)

	for i in range(np.shape(video_noisy)[0]):
		cv2.imshow('frame', video_noisy[i].astype(np.uint8))
		cv2.waitKey(100)

# test_add_noise(sys.argv[1])