import numpy as np
import cv2
cap = cv2.VideoCapture("/media/suresh_arunachalam/user/Project/python_coding/Drone_Detection.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()
ret, current_frame = cap.read()
previous_frame = current_frame
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame = cap.read() 
    fgmask = fgbg.apply(frame)
  #  cv2.resizeWindow("Background Subtraction", 500, 500) 
    cv2.imshow('Background Subtraction',fgmask)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
    #cv2.resizeWindow("Original",  500, 500) 
    cv2.imshow('Original',current_frame)
  #  cv2.resizeWindow("Frame Difference",  500, 500)    
    cv2.imshow('Frame Difference',frame_diff)
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
 #   cv2.imshow('Dense Optical Flow',bgr)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
 
    laplacian = cv2.Laplacian(blurred_frame, cv2.CV_64F)
    canny = cv2.Canny(blurred_frame, 100, 150)
 
    
#    cv2.imshow("Laplacian", laplacian)
   # cv2.resizeWindow("Canny",  500, 500) 
    cv2.imshow("Canny", canny)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    prvs = next
    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()
cap.release()
cv2.destroyAllWindows()
