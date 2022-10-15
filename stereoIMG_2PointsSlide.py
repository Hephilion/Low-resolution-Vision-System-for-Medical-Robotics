# Ctr+K+C - comment CTR+K+U uncomment
from lib2to3.pytree import Base
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import statistics
from mpl_toolkits import mplot3d

import stereo_calibration_f as calib


def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


# Stereo vision setup parameters
calib.stereo_calibration(time_ms=10)

# Reading the mapping values for stereo image rectification
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
trans_vector = cv_file.getNode('trans').mat()
Q = cv_file.getNode('Q').mat()
Baseline = cv_file.getNode('Baseline').real()
focal_length_x = cv_file.getNode('focal_length_x').real()
projMatrixL = cv_file.getNode('projMatrixL').mat()
projMatrixR = cv_file.getNode('projMatrixR').mat()

c_x = projMatrixL[0,2]
c_y = projMatrixL[1,2]
focal_length_y = projMatrixL[1,1]

print(f'Translation Vector = {np.round(trans_vector.transpose(),3)/10} cm')
#Distance between the cameras [cm]
print(f'Baseline = {Baseline*10} mm')
Baseline = Baseline*10
#Camera lense's focal length [pixels]
print(f'Focal Length x- {np.round(focal_length_x,3)} pixels')
# Reprojection Matrix Q
#print(Q.round(1))

## Read image
img_left=cv.imread('images/testLeft/repetP3L1.png')
img_right=cv.imread('images/testRight/repetP3R1.png')

# Grayscale Images
frame_left = cv.cvtColor(img_left,cv.COLOR_BGR2GRAY)
frame_right = cv.cvtColor(img_right,cv.COLOR_BGR2GRAY)

height, width = frame_left.shape[:2]

# cv.imshow("Stereo Pair Unrectified",np.hstack((frame_left, frame_right)))

# while True:
#     key = cv.waitKey(1)
#     if key == ord('q'):
#     #Quit when q is pressed
#         cv.destroyAllWindows()
#         break


####
# Set display window name
cv.namedWindow("Stereo Pair Rectified")
# Variable use to toggle between side by side view and one frame view.
def nothing(x):
    pass
sideBySide = True

cv.namedWindow('ImageSelect')
cv.resizeWindow('ImageSelect',600,40)
cv.createTrackbar('Select','ImageSelect',1,10,nothing)
while True:
    SelectImg = cv.getTrackbarPos('Select','ImageSelect')

    img_left=cv.imread('images/FAR/DPOLeft/oOriL'+str(SelectImg)+'.png')
    img_right=cv.imread('images/FAR/DPORight/oOriR'+str(SelectImg)+'.png')

    # img_left=cv.imread('images/FAR/DOrientationLeft/oDesiredL'+str(SelectImg)+'.png')
    # img_right=cv.imread('images/FAR/DOrientationRight/oDesiredR'+str(SelectImg)+'.png')

    # img_left=cv.imread('images/FAR/DOrientationLeft/oFIML'+str(SelectImg)+'.png')
    # img_right=cv.imread('images/FAR/DOrientationRight/oFIMR'+str(SelectImg)+'.png')

    # img_left=cv.imread('images/FAR/testLeft/repetP1L'+str(SelectImg)+'.png')
    # img_right=cv.imread('images/FAR/testRight/repetP1R'+str(SelectImg)+'.png')

    # img_left=cv.imread('images/FAR/testLeft/testL'+str(SelectImg)+'.png')
    # img_right=cv.imread('images/FAR/testRight/testR'+str(SelectImg)+'.png')

    # img_left=cv.imread('images/FAR/testLeft/FitaL'+str(SelectImg)+'.png')
    # img_right=cv.imread('images/FAR/testRight/FitaR'+str(SelectImg)+'.png'
    

    # img_left=cv.imread('images/FAR/DPositionLeft/PInicioL'+str(SelectImg)+'.png')
    # img_right=cv.imread('images/FAR/DPositionRight/PInicioR'+str(SelectImg)+'.png')

    # img_left=cv.imread('images/FAR/DPositionLeft/PFimL'+str(SelectImg)+'.png')
    # img_right=cv.imread('images/FAR/DPositionRight/PFimR'+str(SelectImg)+'.png')
    
    # Grayscale Images
    frame_left = cv.cvtColor(img_left,cv.COLOR_BGR2GRAY)
    frame_right = cv.cvtColor(img_right,cv.COLOR_BGR2GRAY)
    # Undistort and rectify images
    frame_left_rect = cv.remap(frame_left, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)
    frame_right_rect = cv.remap(frame_right, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)

    img_left_rect = cv.remap(img_left, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)
    img_right_rect = cv.remap(img_right, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)


    if sideBySide: # Show side by side view
        imOut = np.hstack((frame_left_rect, frame_right_rect))
        imOut = cv.putText(imOut, str(SelectImg),(50,55), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
        i = 0
        while i< 480:
            imOut = cv.line(imOut, (0, i),(640*2, i), (0, 0, 255), 2)
            i = i + 30
    else :         # Show overlapping frames
        imOut = np.uint8(frame_left_rect/2 + frame_right_rect/2)
    
    # Display output image
    cv.imshow("Stereo Pair Rectified", imOut)
    # Check for keyboard input
    key = cv.waitKey(1)
    if key == ord('q'):
        # Quit when q is pressed
        cv.destroyAllWindows()
        break
    elif key == ord('t'):
        # Toggle display when t is pressed
        sideBySide = not sideBySide

def nothing(x):
    pass

cv.namedWindow('Parameters')
cv.resizeWindow('Parameters',600,150)

cv.createTrackbar('WinSize','Parameters',10,40,nothing)
cv.createTrackbar('Method','Parameters',3,5,nothing)
cv.createTrackbar('Color','Parameters',3,5,nothing)


cv.namedWindow("Stereo Pair Distance")
mouseX=86; mouseY=158
num_points=0
Points=[]

while True:
    # Image Left and Right Rectified
    imOut = np.hstack((img_left_rect, img_right_rect))
    # Select Left Point
    cv.setMouseCallback("Stereo Pair Distance", mouseCallback)
    # Left Point
    imOut = cv.circle(imOut, (mouseX,mouseY), 1,(0,255,0), 4)
    # Right Line
    imOut = cv.line(imOut, (640, mouseY),(640*2, mouseY), (0, 0, 255), 2)
    imOut = cv.line(imOut, (640, 0),(640, 480), (0, 0, 255), 2)


    # Window Size
    WinSize = cv.getTrackbarPos('WinSize','Parameters')*2+1
    half_WinSize = int(np.floor(WinSize/2))
    imOut = cv.rectangle(imOut,(mouseX-half_WinSize,mouseY-half_WinSize),(mouseX+half_WinSize,mouseY+half_WinSize),(255,0,0),2)
    WinSelect= img_left_rect[mouseY-half_WinSize:mouseY+half_WinSize,mouseX-half_WinSize:mouseX+half_WinSize]
    #cv.imshow("Window around Point",WinSelect)
    # Banda
    imOut = cv.rectangle(imOut,(640,mouseY-half_WinSize),(640*2,mouseY+half_WinSize),(255,0,255),2)
    BandaSelect=img_right_rect[mouseY-half_WinSize:mouseY+half_WinSize,:]
    #cv.imshow("Banda",BandaSelect)

    img2 = BandaSelect.copy()
    w = WinSize;  h=w

    # Methods
    #'cv.TM_CCOEFF',       -> 1
    #'cv.TM_CCOEFF_NORMED' -> 2
    #'cv.TM_CCORR'         -> 3
    #'cv.TM_CCORR_NORMED'  -> 4
    #'cv.TM_SQDIFF'        -> 5
    #'cv.TM_SQDIFF_NORMED' -> 6

    Method = cv.getTrackbarPos('Method','Parameters')
    # Apply Matching
    res = cv.matchTemplate(BandaSelect,WinSelect,Method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if Method in [4,5]:
        x_top_left=min_loc[0]+640
        y_top_left=mouseY-half_WinSize
        top_left = [x_top_left,y_top_left]
    else:
        x_top_left=max_loc[0]+640
        y_top_left=mouseY-half_WinSize
        top_left = [x_top_left,y_top_left]


    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Rectangle around Right Point
    imOut =cv.rectangle(imOut,top_left, bottom_right, (255,0,0), 2)
    ######
    # Color
    Color = cv.getTrackbarPos('Color','Parameters')
    if Color ==0:
        textColor=(0,0,0)
    elif Color ==1:
        textColor=(0,0,255)
    elif Color ==2:
        textColor=(255,0,0)
    elif Color ==3:
        textColor=(0,255,0)
    elif Color ==4:
        textColor=(255,0,255)
    elif Color ==5:
        textColor=(255,255,255)


    # Right Point
    Right_X=round((bottom_right[0]+top_left[0])/2)
    imOut = cv.circle(imOut, (Right_X,mouseY), 1,(0,255,0), 4)
    imOut = cv.putText(imOut, "(" + str(mouseX)+","+str(mouseY)+")",(mouseX-70,mouseY-20-half_WinSize), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    imOut = cv.putText(imOut, "(" + str(Right_X-640)+","+str(mouseY)+")",(Right_X-70,mouseY-20-half_WinSize), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)

    #Depth
    Disparity = Right_X-640-mouseX
    Zglobal = focal_length_x*Baseline/(Disparity)
    Xglobal= Baseline*(mouseX-c_x)/(Disparity)
    Yglobal= Baseline*focal_length_x*(mouseY-c_y)/(Disparity*focal_length_y)

    Z1 = focal_length_x*Baseline/(Disparity-1)
    X1= Baseline*(mouseX-c_x)/(Disparity-1)
    Y1= Baseline*focal_length_x*(mouseY-c_y)/((Disparity-1)*focal_length_y)

    dX = Xglobal - X1
    dY = Yglobal - Y1
    dZ = Zglobal - Z1

    # imOut = cv.putText(imOut, "X = " + str(round(Xglobal,3)) + "mm",(30,30), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    # imOut = cv.putText(imOut, "Y = " + str(round(Yglobal,3)) + "mm",(30,60), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    # imOut = cv.putText(imOut, "Z = " + str(round(Zglobal,3)) + "mm",(30,90), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    # imOut = cv.putText(imOut, "Disp = " + str(round(Disparity,3)) + "pixel",(30,120), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    # imOut = cv.putText(imOut, "dX = " + str(round(dX,3)) + "mm",(30,150), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    # imOut = cv.putText(imOut, "dY = " + str(round(dY,3)) + "mm",(30,180), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    # imOut = cv.putText(imOut, "dZ = " + str(round(dZ,3)) + "mm",(30,210), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)

        # Close window using esc key
    key = cv.waitKey(1)
    if key == ord('q'):
        # Quit when q is pressed
        break
    elif key == ord('s'):
        num_points += 1
        print("Saving Point " + str(num_points))
        # Point - xL xR(+640) y Xglobal Yglobal Zglobal
        point=[mouseX, Right_X, mouseY, Xglobal, Yglobal, Zglobal]
        Points.append(point)
    elif key == ord('p'):
        print('Img = ' + str(SelectImg))
        print(Xglobal,Yglobal,Zglobal )


    #Draw left and right saved points
    for num in Points:
        imOut = cv.circle(imOut, (num[0],num[2]), 2,(0,0,255), 4)
        imOut = cv.circle(imOut, (num[1],num[2]), 2,(0,0,255), 4)

    #imOut = cv.putText(imOut, "Press 's' to save the point",(30,450), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
    cv.imshow("Stereo Pair Distance",imOut)

cv.destroyAllWindows()

##################################################################################################
# Show Image with the points
##################################################################################################
while True:
    # Image Left and Right Rectified
    imOut = np.hstack((img_left_rect, img_right_rect))

    pontos=Points
    for num in pontos:
        imOut = cv.circle(imOut, (num[0],num[2]), 2,(0,0,255), 4)
        imOut = cv.circle(imOut, (num[1],num[2]), 2,(0,0,255), 4)

    # Point - xL xR(+640) y Xglobal Yglobal Zglobal
    p0, p1  = Points
    xL0, xR0, y0, Xglobal0, Yglobal0, Zglobal0 = p0
    xL1, xR1, y1, Xglobal1, Yglobal1, Zglobal1 = p1
    
    # Draw line between the 2 points
    imOut = cv.line(imOut,(xL0,y0),(xL1,y1),color=(255,255,0),thickness=2)
    imOut = cv.line(imOut,(xR0,y0),(xR1,y1),color=(255,255,0),thickness=2)

    if xL0<xL1:
        sinal = -1
    else:
        sinal= 1

    # Draw Triangle at the end of vectors
    pt1 = (xL1, y1);    pt2 = (xL1+sinal*7, y1+7);    pt3 = (xL1+sinal*7, y1-7)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    #imOut = cv.drawContours(imOut, [triangle_cnt], 0, (0,255,0), -1)

    pt1 = (xR1, y1);    pt2 = (xR1+sinal*7, y1+7);    pt3 = (xR1+sinal*7, y1-7)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    #imOut = cv.drawContours(imOut, [triangle_cnt], 0, (0,255,0), -1)

    cv.imshow("Stereo Pair Distance",imOut)

    # Close window using esc key
    key = cv.waitKey(1)
    if key == ord('q'):
        print()
        # Quit when q is pressed
        break

cv.destroyAllWindows()


