import cv2

video_file = "bus.y4m"
cap = cv2.VideoCapture(video_file)

frameCount = 0
while(cap.isOpened()):
	ret, frame = cap.read()
	frameCount +=1
	if ret:
		cv2.imshow('frame', frame)
	else:
		break
	cv2.waitKey(100)
		
print frameCount
cap.release()
cv2.destroyAllWindows()
