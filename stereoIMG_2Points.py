# Ctr+K+C - comment CTR+K+U uncomment
from cmath import pi
from lib2to3.pytree import Base
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import statistics
from mpl_toolkits import mplot3d
import math
import stereo_calibration_f as calib
#import depthMap as Map

def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


# Stereo vision setup parameters
#calib.stereo_calibration(time_ms=100)

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
focal_length = cv_file.getNode('focal_length').real()
projMatrixL = cv_file.getNode('projMatrixL').mat()
projMatrixR = cv_file.getNode('qprojMatrixR').mat()


print(f'Translation Vector = {np.round(trans_vector.transpose(),3)/10} cm')
#Distance between the cameras [cm]
print(f'Baseline = {Baseline*10} mm')
Baseline = Baseline*10
#Camera lense's focal length [pixels]
print(f'Focal Length - {np.round(focal_length,3)} pixels')
# Reprojection Matrix Q
#print(Q.round(1))

## Read image
img_left=cv.imread('images/MID/testLeft/repetP1L0.png')
img_right=cv.imread('images/MID/testRight/repetP1R0.png')

# Grayscale Images
frame_left = cv.cvtColor(img_left,cv.COLOR_BGR2GRAY)
frame_right = cv.cvtColor(img_right,cv.COLOR_BGR2GRAY)

height, width = frame_left.shape[:2]

cv.imshow("Stereo Pair Unrectified",np.hstack((frame_left, frame_right)))

while True:
    key = cv.waitKey(1)
    if key == ord('q'):
    #Quit when q is pressed
        cv.destroyAllWindows()
        break


# Undistort and rectify images
frame_left_rect = cv.remap(frame_left, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)
frame_right_rect = cv.remap(frame_right, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)

img_left_rect = cv.remap(img_left, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)
img_right_rect = cv.remap(img_right, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT,0)

####
# Set display window name
cv.namedWindow("Stereo Pair Rectified")
# Variable use to toggle between side by side view and one frame view.
sideBySide = True
while True:
    if sideBySide: # Show side by side view
        imOut = np.hstack((frame_left_rect, frame_right_rect))
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
cv.resizeWindow('Parameters',600,600)

cv.createTrackbar('WinSize','Parameters',30,40,nothing)
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
    if Method in [5,6]:
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
    Zglobal = focal_length*Baseline/(Disparity)
    Xglobal= Zglobal*mouseX/focal_length
    Yglobal= Zglobal*mouseY/focal_length
    Z1 = focal_length*Baseline/(Disparity+1)
    X1= Baseline*mouseX/(Disparity+1)
    Y1= Baseline*mouseY/(Disparity+1)
    dX = Xglobal - X1
    dY = Yglobal - Y1
    dZ = Zglobal - Z1

    imOut = cv.putText(imOut, "X = " + str(round(Xglobal,3)) + "mm",(30,30), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    imOut = cv.putText(imOut, "Y = " + str(round(Yglobal,3)) + "mm",(30,60), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    imOut = cv.putText(imOut, "Z = " + str(round(Zglobal,3)) + "mm",(30,90), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    imOut = cv.putText(imOut, "Disp = " + str(round(Disparity,3)) + "pixel",(30,120), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    imOut = cv.putText(imOut, "dX = " + str(round(dX,3)) + "mm",(30,150), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    imOut = cv.putText(imOut, "dY = " + str(round(dY,3)) + "mm",(30,180), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)
    imOut = cv.putText(imOut, "dZ = " + str(round(dZ,3)) + "mm",(30,210), cv.FONT_HERSHEY_SIMPLEX, 0.9, textColor,2)

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
        print(Xglobal,Yglobal,Zglobal )


    #Draw left and right saved points
    for num in Points:
        imOut = cv.circle(imOut, (num[0],num[2]), 2,(0,0,255), 4)
        imOut = cv.circle(imOut, (num[1],num[2]), 2,(0,0,255), 4)

    imOut = cv.putText(imOut, "Press 's' to save the point",(30,450), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
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
    imOut = cv.drawContours(imOut, [triangle_cnt], 0, (0,255,0), -1)

    pt1 = (xR1, y1);    pt2 = (xR1+sinal*7, y1+7);    pt3 = (xR1+sinal*7, y1-7)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    imOut = cv.drawContours(imOut, [triangle_cnt], 0, (0,255,0), -1)

    cv.imshow("Stereo Pair Distance",imOut)

    # Close window using esc key
    key = cv.waitKey(1)
    if key == ord('q'):
        print()
        # Quit when q is pressed
        break

cv.destroyAllWindows()
#######################################################################################################
# Transformation Matrix from CAM to KUKA + Translation vector
#######################################################################################################
R_CAM_KUKA = np.zeros((3,3))
R_CAM_KUKA[0,:] = [ 0.0602489280055207	,-0.160734788868569	,0.985157040436478]
R_CAM_KUKA[1,:] = [-0.997038712768600	,0.0375614536164914	,0.0671039674158612]
R_CAM_KUKA[2,:] = [-0.0477898725141467	,-0.986282649573433	,-0.157995769675929]

#Trans_CAM_KUKA = np.array([[Xglobal0],  [Yglobal0], [Zglobal0]])
#######################################################################################################
# Transform X,Y,Z of the 2 points from CAM ref to KUKA ref
# Compute vector Z to be the vector made by the points
# Get X and Y vector by cross product
# Normalize vectors
# Construct Rotation matrix in the KUKA ref frame
#######################################################################################################
# 2 Points in the CAM frame
PCam0 = np.array([[Xglobal0],  [Yglobal0], [Zglobal0]]); PCam1 = np.array([[Xglobal1],  [Yglobal1], [Zglobal1]]);

#  Transform from Cam ref to KUKA ref 
PKuka0 = np.dot(R_CAM_KUKA,PCam0);      PKuka1 = np.dot(R_CAM_KUKA,PCam1); 

# Construct Z vector in KUKA ref

ZvecKUKA = (PKuka1 - PKuka0).T # Row Vector
Z_squared = [z ** 2 for z in ZvecKUKA]
Z_normalized = ZvecKUKA / np.sqrt(np.sum(Z_squared))

XvecKUKA = np.cross([0,1,0],Z_normalized)
X_squared = [x ** 2 for x in XvecKUKA]
X_normalized = XvecKUKA / np.sqrt(np.sum(X_squared))

YvecKUKA = np.cross(X_normalized,Z_normalized)
Y_squared = [y ** 2 for y in YvecKUKA]
Y_normalized = YvecKUKA / np.sqrt(np.sum(Y_squared)) 

# RotInKUKA = [X]
#     [Y]
#     [Z]
# Rot Matrix in Camera Ref Frame
Rot_matrix = np.zeros((3,3))
Rot_matrix = np.array([[X_normalized][0][0],[Y_normalized][0][0], [Z_normalized][0][0]])
Rot_matrix = Rot_matrix.T
# RotInKUKA = [X    ; Y  ;   Z  ]

rot_sin = np.sin(np.pi)
rot_cos = np.cos(np.pi)

RotY = np.stack([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
Rot_matrix = np.dot(Rot_matrix,RotY)
print(Rot_matrix)

###########################################################################################
# Rotation Matrix to Euler angles
#-------------------------------------------------------------------------------------------
# Rotation of ψ radians about the x-axis  
# Rotation of θ radians about the y-axis
# Rotation of φ radians about the z-axis
############################################################################################












##################################################################################################
# Checks if a matrix is a valid rotation matrix.
Rt = np.transpose(Rot_matrix)
shouldBeIdentity = np.dot(Rt, Rot_matrix)
I = np.identity(3, dtype = Rot_matrix.dtype)
n = np.linalg.norm(I - shouldBeIdentity)
if n >= 1e-6:
    print('Not valid rotation matrix')
elif n < 1e-6:
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

    sy = math.sqrt(Rot_matrix[0,0] * Rot_matrix[0,0] +  Rot_matrix[1,0] * Rot_matrix[1,0])
    singular = sy < 1e-6

    if  not singular : 
        x_angle = math.atan2(Rot_matrix[2,1] , Rot_matrix[2,2])
        y_angle = math.atan2(-Rot_matrix[2,0], sy)
        z_angle = math.atan2(Rot_matrix[1,0], Rot_matrix[0,0])
    else :
        x_angle = math.atan2(-Rot_matrix[1,2], Rot_matrix[1,1])
        y_angle = math.atan2(-Rot_matrix[2,0], sy)
        z_angle = 0
    
    np.array([x_angle, y_angle, z_angle])
print('X angle - ' + str(x_angle))
print('Y angle - ' + str(y_angle))
print('Z angle - ' + str(z_angle))
print('')
print('X angle º- ' + str(x_angle*180/np.pi))
print('Y angle º- ' + str(y_angle*180/np.pi))
print('Z angle º- ' + str(z_angle*180/np.pi))