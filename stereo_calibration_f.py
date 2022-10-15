import cv2 as cv
import numpy as np
import glob
def stereo_calibration(time_ms):
    ## FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS
    print('Starting Calibration!!')

    chessboardSize = (9,6) # size of chessboard 
    frameSize = (640,480) # Depends on resolution

    # termination criteria - default by openCV
    criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,60,0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    # Check size of chessboard square
    size_of_chessboard_squares_mm = 16.5#15.5 #22
    objp = objp * size_of_chessboard_squares_mm

    # Reads all images in files
    imagesLeft = sorted(glob.glob('images/FAR2/stereoLeft/*.png'))
    imagesRight = sorted(glob.glob('images/FAR2/stereoRight/*.png'))

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = [] # 2d points in image plane.

    

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):

        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        # Convert to grayscale
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:

            objpoints.append(objp)

            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv.imshow('img left', imgL)
            cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv.imshow('img right', imgR)
            cv.waitKey(time_ms)


    cv.destroyAllWindows()


    ############## CALIBRATION #######################################################
    # Calib succesful, Cam Matrix, distortion,rotation, translation
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    
    # Generate a new optimal Camera Matrix
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
    fov_x_left, fov_y_left, focal_len_left, principal, aspect = cv.calibrationMatrixValues(cameraMatrixL,(widthL, heightL),apertureWidth=15,apertureHeight=15)
    print()
    print('FOV X:',fov_x_left)
    print('FOV Y:',fov_y_left)
    print('Focal Length Left:',focal_len_left)
    print('RMSE Left:',retL)

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))
    fov_x_rigth, fov_y_right, focal_len_right, principal, aspect = cv.calibrationMatrixValues(cameraMatrixR,(widthR, heightR),apertureWidth=15,apertureHeight=15)
    print('RMSE Right:',retR)
    ########## Stereo Vision Calibration #############################################

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Essential Mat and Fundamental Mat are calculated.
    # Hence intrinsic parameters are the same 

    criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    print('RMSE StereoPair:',retStereo)
    #print('Translation Vector - {}',trans.transpose())
    ########## Stereo Rectification #################################################

    rectifyScale = 1
    flags |= cv.CALIB_ZERO_DISPARITY
    #flags |= 0
    #rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans,rectifyScale,(0,0))
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans,rectifyScale,flags=flags)
    #rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot.T, -trans,flags,rectifyScale,(0,0))
    
    # print('\n Projection Matrix Left')
    # print(projMatrixL)
    # print('\n Projection Matrix Right')
    # print(projMatrixR)
    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

    print("Saving parameters!")
    #print(projMatrixL)

    # Baseline 
    Baseline =np.linalg.norm(trans/10)

    # Focal Length
    focal_length_x = projMatrixL[0][0]
    focal_length_y = projMatrixL[1][1]
    c_x_left = projMatrixL[0][2]
    c_y_left = projMatrixL[1][2]

    cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])
    cv_file.write('Baseline',Baseline)
    cv_file.write('focal_length_x',focal_length_x)
    cv_file.write('focal_length_y',focal_length_y)
    cv_file.write('c_x_left',c_x_left)
    cv_file.write('c_y_left',c_y_left)
    cv_file.write('trans',trans)
    cv_file.write('Q',Q)
    cv_file.write('projMatrixL',projMatrixL)
    cv_file.write('projMatrixR',projMatrixR)

    cv_file.release()

    blackimg = np.zeros((300,480,3))
    
    # while True:
    #     colorText = (0,255,0)
    #     cv.putText(blackimg, "RMSE Left = " + str(np.round(retL,6)) ,(30,30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(blackimg, "RMSE Right = " + str(np.round(retR,6)) ,(30,60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(blackimg, "RMSE Stereo = " + str(np.round(retStereo,6)) ,(30,90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(blackimg, "FOV X Left= " + str(np.round(fov_x_left,6)) ,(30,120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(blackimg, "FOV Y Left= " + str(np.round(fov_y_left,6)) ,(30,150), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(blackimg, "FOV X Right= " + str(np.round(fov_x_rigth,6)) ,(30,180), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(blackimg, "FOV Y Right= " + str(np.round(fov_y_right,6)) ,(30,210), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
        
    
    #     cv.imshow('STATS',blackimg)
    
    
    #     key = cv.waitKey(1)
    #     if key == ord('q'):
    #         # Quit when q is pressed
    #         cv.destroyAllWindows()
    #         break
    


    # InstrinsicStats = np.zeros((600,480,3))
    
    # while True:
    #     colorText = (0,255,0)
    #     a_text = 0;
    #     # Instrinsics Left
    #     cv.putText(InstrinsicStats, "f_x_Left = " + str(np.round(cameraMatrixL[0,0],3)) +'pixels' ,(30,a_text + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "f_y_Left = " + str(np.round(cameraMatrixL[1,1],3)) + 'pixels' ,(30,a_text + 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "c_x_Left = " + str(np.round(cameraMatrixL[0,2],3)) +'pixels' ,(30,a_text + 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "c_y_Left = " + str(np.round(cameraMatrixL[1,2],3)) + 'pixels' ,(30,a_text + 120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.line(InstrinsicStats,(0,130),(500,a_text + 130),colorText)

    #     # Distortion Left
    #     a_text = a_text + 130;
    #     cv.putText(InstrinsicStats, "k1_Left = " + str(np.round(distL[0,0],3)) +'pixels' ,(30,a_text + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "k2_Left = " + str(np.round(distL[0,1],3)) + 'pixels' ,(30,a_text + 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "k3_Left = " + str(np.round(distL[0,4],3)) +'pixels' ,(30,a_text + 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "p1_Left = " + str(np.round(distL[0,2],3)) +'pixels' ,(30,a_text + 120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "p2_Left = " + str(np.round(distL[0,3],3)) + 'pixels' ,(30,a_text + 150), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.line(InstrinsicStats,(0,a_text + 160),(500,a_text + 160),colorText)

    #     # Instrinsics Right
    #     a_text = a_text + 160;
    #     cv.putText(InstrinsicStats, "f_x_Right = " + str(np.round(cameraMatrixR[0,0],3)) +'pixels' ,(30,a_text + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "f_y_Right = " + str(np.round(cameraMatrixR[1,1],3)) + 'pixels' ,(30,a_text + 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "c_x_Right = " + str(np.round(cameraMatrixR[0,2],3)) +'pixels' ,(30,a_text + 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "c_y_Right = " + str(np.round(cameraMatrixR[1,2],3)) + 'pixels' ,(30,a_text + 120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.line(InstrinsicStats,(0,a_text + 130),(500,a_text + 130),colorText)

    #     # Distortion Right
    #     a_text = a_text + 130;
    #     cv.putText(InstrinsicStats, "k1_Left = " + str(np.round(distR[0,0],3)) +'pixels' ,(30,a_text + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "k2_Left = " + str(np.round(distR[0,1],3)) + 'pixels' ,(30,a_text + 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "k3_Left = " + str(np.round(distR[0,4],3)) +'pixels' ,(30,a_text + 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "p1_Left = " + str(np.round(distR[0,2],3)) +'pixels' ,(30,a_text + 120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(InstrinsicStats, "p2_Left = " + str(np.round(distR[0,3],3)) + 'pixels' ,(30,a_text + 150), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)


    #     cv.imshow('IntrinsicSTATS',InstrinsicStats)
    
    
    #     key = cv.waitKey(1)
    #     if key == ord('q'):
    #         # Quit when q is pressed
    #         cv.destroyAllWindows()
    #         break
    
    # ExtrinsicStats = np.zeros((600,480,3))
    
    # while True:
    #     colorText = (0,255,0)
    #     a_text = 0;
    #     # Extrinsics
    #     cv.putText(ExtrinsicStats, "R = " ,(30,a_text + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(rot[0,:],3)) ,(30,a_text + 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(rot[1,:],3)) ,(30,a_text + 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(rot[2,:],3)) ,(30,a_text + 120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
        
    #     cv.putText(ExtrinsicStats, "T = " ,(30,a_text + 180), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(trans[0],3)) ,(30,a_text + 210), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(trans[1],3)) ,(30,a_text + 240), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(trans[2],3)) ,(30,a_text + 270), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
        
    #     cv.putText(ExtrinsicStats, "Baseline = " + str(np.round(Baseline*10,3)) +'mm' ,(30,310), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
        
    #     cv.putText(ExtrinsicStats, "F = " ,(30,370), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(fundamentalMatrix[0,:],3)) ,(30,400), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(fundamentalMatrix[1,:],3)) ,(30,430), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ExtrinsicStats, str(np.round(fundamentalMatrix[2,:],3)) ,(30,460), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)

    #     cv.imshow('ExtrinsicSTATS',ExtrinsicStats)
    
    
    #     key = cv.waitKey(1)
    #     if key == ord('q'):
    #         # Quit when q is pressed
    #         cv.destroyAllWindows()
    #         break

    # ProjectionMatrices = np.zeros((400,880,3))
    
    # while True:
    #     colorText = (0,255,0)
    #     a_text = 0;
    #     # Extrinsics
    #     cv.putText(ProjectionMatrices, "Projection M Left = " ,(30,a_text + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ProjectionMatrices, str(np.round(projMatrixL[0,:],3)) ,(30,a_text + 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ProjectionMatrices, str(np.round(projMatrixL[1,:],3)) ,(30,a_text + 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ProjectionMatrices, str(np.round(projMatrixL[2,:],3)) ,(30,a_text + 120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)

    #     a_text = a_text + 150;
    #     cv.putText(ProjectionMatrices, "Projection M Right = " ,(30,a_text + 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ProjectionMatrices, str(np.round(projMatrixR[0,:],3)) ,(30,a_text + 60), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ProjectionMatrices, str(np.round(projMatrixR[1,:],3)) ,(30,a_text + 90), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
    #     cv.putText(ProjectionMatrices, str(np.round(projMatrixR[2,:],3)) ,(30,a_text + 120), cv.FONT_HERSHEY_SIMPLEX, 0.9, colorText,2)
        

    #     cv.imshow('ProjectionMatrices',ProjectionMatrices)
    
    
    #     key = cv.waitKey(1)
    #     if key == ord('q'):
    #         # Quit when q is pressed
    #         cv.destroyAllWindows()
    #         break    
    return 
