import cv2 as cv
import numpy as np

# Open webcams
cam = cv.VideoCapture(2) # Left
cam2= cv.VideoCapture(1) # Right

num=0
while cam.isOpened():
    # Read each individual frame from webcam
    succes1, img = cam.read()
    succes2, img2 = cam2.read()

    k = cv.waitKey(5) 

    if k ==27:
        break
    # Press 's' to capture image
    elif k == ord('s'): # wait for 's' key to save and exit
        
        #########################################################################
        #num=10
        if num == 0:
            print()
            cv.imwrite('images/FAR2/testLeft/testL' + str(num) + '.png',img)
            cv.imwrite('images/FAR2/testRight/testR' + str(num) + '.png',img2)
            
            # cv.imwrite('images/grelhaLeft/grelhaL' + str(num) + '.png',img)
            # cv.imwrite('images/grelhaRight/grelhaR' + str(num) + '.png',img2)
        else:
            cv.imwrite('images/FAR2/stereoLeft/imgL' + str(num) + '.png',img)
            cv.imwrite('images/FAR2/stereoRight/imgR' + str(num) + '.png',img2)

        print('Images Saved!! Total='+ str(num))
        num +=1
        
    cv.imshow('Left', img)
    cv.imshow('Right', img2)
# Press ESC to exit
# Release and destroy all windows before termination
cam.release()
cam2.release()

cv.destroyAllWindows()

