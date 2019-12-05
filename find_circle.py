import cv2
import numpy as np
import os
import time

stage = 3
send_info_circle = [5]*5
def method_HSV(img):
    res = img.copy()
    if stage == 1: #green
        lower = np.array([50, 50, 180])
        upper = np.array([80, 180, 255])
    if stage == 2: #blue
        lower = np.array([100, 80, 220])
        upper = np.array([130, 255, 255]) #124max
    if stage == 3: #red #have another value
        lower = np.array([0, 100, 150])
        upper = np.array([7, 200, 220])

        lower2 = np.array([156, 100, 150])
        upper2 = np.array([180, 200, 220])
        # lower = np.array([0, 140, 150])
        # upper = np.array([7, 200, 220])
        # lower = np.array([0, 130, 135])
        # upper = np.array([15, 200, 220])  
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_mask = cv2.inRange(hsv, lower, upper)
    if stage == 3:
        lower2 = np.array([156, 100, 150])
        upper2 = np.array([180, 200, 220])
        hsv2 = cv2.inRange(hsv, lower2, upper2)
        # cv2.imshow("hsv2",hsv2)
        # if hsv2 is not None :
        hsv_mask = cv2.bitwise_or(hsv_mask, hsv2)
        # cv2.imshow("hsv_all", hsv2)
    res = cv2.bitwise_and(img, img, mask=hsv_mask)
    return res

def method_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if stage == 1:
        ret,cc = cv2.threshold(gray,205,255,0)
    if stage == 2:
        ret,cc = cv2.threshold(gray,0,255,0)
    if stage == 3:
        ret,cc = cv2.threshold(gray,80,255,0)
    return cc

def method_contours(img,ori):
    binary, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    if len(contours)>0:
        max_idx = np.argmax(np.array(area))  #max contours
        x, y, w, h = cv2.boundingRect(contours[max_idx])
        if w * h == 0 or w / h < 0.3 or w / h < 0.3 or w * h < 4000:
            print("area = ",w * h)
            x=0; y=0; w=0; h = 0
            return ori, x, y, w, h  # [stage,midx,midy,width,height]
        else:
            color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            res3 = cv2.rectangle(ori, (x, y), (x + w, y + h), (0, 0, 255), 2)
            res4 = cv2.circle(res3,(int(x + w/2), int(y + h/2)), 5, (0, 255, 255), 2)
            return res4,int(x + w/2), int(y + h/2),w,h
    else:
        print("no contours!")
        return ori,0, 0,0,0
def process(img):
    shape = img.shape
    global send_info_circle
    flag = 0
    hsv = method_HSV(img)
    if hsv is None:
        print("no hsv value")
        flag = 1
        return img,flag
    cv2.imshow("hsv",hsv)
    threshold = method_threshold(hsv)
    if  threshold is None:
        print("no threshold value")
        flag = 1
        return img, flag
    cv2.imshow("threshold",threshold)

    kernel1 = np.ones((3, 3), np.uint8)
    threshold = cv2.dilate(threshold,kernel1,iterations=1)
    kernel2 = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(threshold,kernel2,iterations = 2)
    if dilate is None:
        print("no dilate value")
        flag = 1
        return img, flag
    cv2.imshow("dilate", dilate)
    contours,midx,midy,width,height = method_contours(dilate, img)
    if  width*height == 0:
        send_info_circle = [0, 0, 0, 0, 0]
    else:
        send_info_circle = [stage,midx-shape[1]/2,midy-shape[0]/2,width,height] #height = img.shape[0] width = img.shape[1]
    return contours,flag

cnt=1
save_path = "E:\\myvcwork\\repos\\fuzajidian\\1115pic\\"
def save_img(img):
    global cnt
    if (cnt%30 ==0):
        name = save_path +str(cnt)+'.jpg'
        cv2.imwrite(name,img)
    cnt+=1
    return img


if __name__ == '__main__':
    video = True
    if video:
        if stage == 1:
            video_path = r"E:\myvcwork\repos\fuzajidian\fuzajidian\1115green.mp4"
        if stage == 2:
            video_path = r"E:\myvcwork\repos\fuzajidian\fuzajidian\1115blue.mp4"
        if stage == 3:
            video_path = r"E:\myvcwork\repos\fuzajidian\fuzajidian\1115red.mp4"
        cameraCapture = cv2.VideoCapture(video_path)
        cv2.namedWindow("MyWindow",0)
        success, frame = cameraCapture.read()
        while  success:
            start = time.time()

            ori= frame.copy()
            #ori = cv2.resize(ori,(1080,720))

            end_img, flag = process(ori)
            # try:
            #     end_img,flag = process(ori)
            #     if flag == 1:
            #         print("end_img == 0")
            #         continue
            # except:
            #     print("error occured")

            end = time.time()
            seconds = end - start
            fps = 1 / seconds

            cv2.putText(end_img, str(fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 2)
            cv2.imshow("MyWindow", end_img)
            cv2.waitKey(5)
            success, frame = cameraCapture.read()
    else:
        path = "E:\\myvcwork\\repos\\fuzajidian\\picc\\43.jpg"
        frame = cv2.imread(path)
        if len(frame)==0:
            print("no data")
            cv2.waitKey()
        else:
            end = process(frame)
            cv2.imshow("MyWindow", end)
            cv2.waitKey()