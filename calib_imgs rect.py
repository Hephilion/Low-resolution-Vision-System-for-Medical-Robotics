import cv2 as cv
import numpy as np

# Open webcams
cam = cv.VideoCapture(0) # Left
cam2= cv.VideoCapture(2) # Right

# Reading the mapping values for stereo image rectification
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


num=0
while cam.isOpened():
    # Read each individual frame from webcam
    succes1, img = cam.read()
    succes2, img2 = cam2.read()

    # Undistort and Rectify Image    
    frame_left_rect = cv.remap(img, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    frame_right_rect = cv.remap(img2, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    imOut = np.hstack((frame_left_rect, frame_right_rect))
    #print(img.shape)

    k = cv.waitKey(5) 

    if k ==27:
        break
    # Press 's' to capture image
    elif k == ord('s'): # wait for 's' key to save and exit
        # cv.imwrite('images/stereoLeft/imgL' + str(num) + '.png',img)
        # cv.imwrite('images/stereoRight/imgR' + str(num) + '.png',img2)
        #########################################################################
        #num=10
        # cv.imwrite('images/FAR2/testLeft/repetP8L' + str(num) + '.png',img)
        # cv.imwrite('images/FAR2/testRight/repetP8R' + str(num) + '.png',img2)
        # print('Images Saved!! P8 Total='+ str(num))


        # ### Desired Orientation
        num = 1

        cv.imwrite('images/FAR/DPOLeft/oInicioL' + str(num) + '.png',img)
        cv.imwrite('images/FAR/DPORight/oInicioR' + str(num) + '.png',img2)

        # cv.imwrite('images/FAR/DOrientationLeft/oDesiredL' + str(num) + '.png',img)
        # cv.imwrite('images/FAR/DOrientationRight/oDesiredR' + str(num) + '.png',img2)

        # cv.imwrite('images/FAR/DOrientationLeft/oInicioL' + str(num) + '.png',img)
        # cv.imwrite('images/FAR/DOrientationRight/oInicioR' + str(num) + '.png',img2)

        # cv.imwrite('images/FAR/DOrientationLeft/oFimL' + str(num) + '.png',img)
        # cv.imwrite('images/FAR/DOrientationRight/oFimR' + str(num) + '.png',img2)

        # cv.imwrite('images/FAR/DOrientationLeft/oRotYL' + str(num) + '.png',img)
        # cv.imwrite('images/FAR/DOrientationRight/oRotYR' + str(num) + '.png',img2)

        # cv.imwrite('images/FAR/DOrientationLeft/oRotZL' + str(num) + '.png',img)
        # cv.imwrite('images/FAR/DOrientationRight/oRotZR' + str(num) + '.png',img2)
        
        print('Images Saved!! Total='+ str(num))
        num +=1
        
    # cv.imshow('Left', img)
    # cv.imshow('Right', img2)
    cv.imshow('IMAGE RECTIFIED', imOut)
# Press ESC to exit
# Release and destroy all windows before termination
cam.release()
cam2.release()

cv.destroyAllWindows()

