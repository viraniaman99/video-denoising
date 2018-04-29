import cv2
import numpy as np
import sys

video_file = sys.argv[1]
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

cap.release()
cv2.destroyAllWindows()


video = np.asarray(video, dtype = np.float32)
frameCount = np.shape(video)[0]
M, N = np.shape(video)[1], np.shape(video)[2]
nchannels = np.shape(video)[3]
print "----- Video Details ---- "
print "# Frames: ",  frameCount
print "Image Size", M,"X",N
print "# Channels: ", nchannels
