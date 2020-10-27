import cv2
import time
import numpy as np


# Setting up the detector and the matcher..
detector = cv2.xfeatures2d.SIFT_create()  #Here we use SIFT detector for better accuracy

#For SIFT the following parameters are recomended for FLANN based matcher
FLANN_INDEX_KDTREE = 0
indexParam = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
search_params = dict(checks=500)   #Higher value gives better accuracy but it will take more time to compute
flann = cv2.FlannBasedMatcher(indexParam, search_params)


cam = cv2.VideoCapture(0)

#Naming the Final window
windowname = 'Result'
cv2.namedWindow(windowname)

MIN_MATCH_COUNT = 30  # Change the value as per your requrement, otherwise just keep it as it is.

# Define the VideoWritter Parameters
vidSize = (int(cam.get(3)), int(cam.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.mp4', fourcc, 12.0, vidSize)


#Process the Cropped image from the frame
def get_target_data(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des


# Correcting the selected target object origin
def correct_origins(x, y, p, q):
    if x > p:
        temp = x
        x = p
        p = temp
    if y > q:
        temp = y
        y = q
        q = temp

    if x < 0: x = 0
    if y < 0: y = 0

    return x, y, p, q


# Set up mouse event function
drawBorder = False
canMatch = False
pause = False
def click_event(event, x, y, flags, param):
    global sx, sy, ex, ey, des1, kp1, cropped, canMatch, drawBorder, pause

    if event == cv2.EVENT_LBUTTONDOWN:
        sx = x
        sy = y
        ex = x
        ey = y
        eraseRectangle = True
        drawBorder = True
        pause = True

    if event == cv2.EVENT_LBUTTONUP:
        ex = x
        ey = y
        sx, sy, ex, ey = correct_origins(sx, sy, ex, ey)
        drawBorder = False
        pause = False

        #crop the selected area from the main frame and process it
        if sx != ex and sy != ey:
            cropped = frame[sy:ey, sx:ex]
            cropped, kp1, des1 = get_target_data(cropped)
            canMatch = True
            cv2.imshow('cropped', cropped)

    if event == cv2.EVENT_MOUSEMOVE:
        if drawBorder == True:
            if sx == x and sy == y:
                drawBorder = False
            ex = x
            ey = y


cv2.setMouseCallback(windowname, click_event)  # Start the mouse event

while True:
    startTime = time.time()
    if pause == False:
        _, frame = cam.read()
    frame2 = frame.copy()
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = detector.detectAndCompute(img2, None)    #detect the features of the captured frame

    if drawBorder == True:
        cv2.rectangle(frame2, (sx, sy), (ex, ey), (0, 255, 255), 2)

    if canMatch == True:
        try:
            matches = flann.knnMatch(des2, des1, k=2)  #matching the features between the captured frame and the object
            goodMatches = []

            for m, n in matches:
                if m.distance < n.distance * 0.7: #Collect only the good features that are matched 70%
                    goodMatches.append(m)

            if len(goodMatches) >= MIN_MATCH_COUNT:
                train_pts = []
                quary_pts = []
                for m in goodMatches:
                    train_pts.append((kp1[m.trainIdx].pt))
                    quary_pts.append((kp2[m.queryIdx].pt))

                train_pts, quary_pts = np.float32((train_pts, quary_pts))
                H, status = cv2.findHomography(train_pts, quary_pts, cv2.RANSAC, 5.0)   #Find the perspective transfrom of the object
                h, w = cropped.shape
                trainingBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
                queryBorder = cv2.perspectiveTransform(trainingBorder, H)
                if drawBorder == False:
                    cv2.polylines(frame2, [np.int32(queryBorder)], True, (0, 255, 255), 3)  #Drawing the border around the object in the captured frame

            cv2.putText(frame2, str(len(goodMatches)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   #Print text on the frame of the total number of matched featues

        except Exception as e:
            pass
        
    #Writing the final video file
    out.write(frame2)
    
    endTime = time.time()

    cv2.putText(frame2, ('fps=' + str(int(1/(endTime - startTime)))), (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)  #print fps on the frame

    cv2.imshow(windowname, frame2)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
out.release()
cam.release()
