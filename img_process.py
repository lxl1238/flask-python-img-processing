from collections import OrderedDict
import numpy as np
import dlib
import cv2
import math
import random
import string

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))])

def get_test(imgName):
    detector = dlib.get_frontal_face_detector()
    # 获取人脸检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    image = cv2.imread('static/images/'+imgName+'.jpg')
    img=cv2.resize(image,dsize=(1100,780),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)

    img_filtering = cv2.bilateralFilter(img,5,25,50)
    # cv2.imshow('bilateralFilter',img_filtering)
    # cv2.waitKey()

    img_gray = cv2.cvtColor(img_filtering, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',img_gray)
    # cv2.waitKey()

    def shape_to_np(shape, dtype="int"):
        # 创建68*2
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        # 遍历每一个关键点
        # 得到坐标
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    rects = detector(img_gray, 0)
    for rect in rects:
        shape = predictor(img_gray, rect)
        shape = shape_to_np(shape)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(img_filtering, [leftEyeHull], -1, (0, 255, 0), 2)
    cv2.drawContours(img_filtering, [rightEyeHull], -1, (0, 255, 0), 2)
    # cv2.imshow('result',img_filtering)
    # cv2.waitKey()

    # print(leftEye[0])
    # print(leftEye[3])
    # print(rightEye[0])
    # print(rightEye[3])

    # dets = detector(img_gray, 1)
    # for face in dets:
    # 	shape = predictor(img_filtering, face)  # 寻找人脸的68个标定点
    # 	# 遍历所有点，打印出其坐标，并圈出来
    # 	for pt in shape.parts():
    # 		pt_pos = (pt.x, pt.y)
    # 		cv2.circle(img_filtering, pt_pos, 2, (0, 255, 0), 1)
    # 	cv2.imshow("image", img_filtering)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    left=(leftEye[0]+leftEye[3])/2
    right=(rightEye[0]+rightEye[3])/2
    print((leftEye[0]+leftEye[3])/2)
    print((rightEye[0]+rightEye[3])/2)

    cv2.circle(img_filtering, (int(left[0]),int(left[1])), 1, (0,0,255),2)
    cv2.circle(img_filtering, (int(right[0]),int(right[1])), 1, (0,0,255),2)
    # cv2.imshow('result',img_filtering)
    # cv2.waitKey()


    sub=right-left
    PD_pixel_distance=math.hypot(sub[0],sub[1])
    value = 9.83
    print('瞳孔的像素距离 = ',PD_pixel_distance)

    cell_size=20
    (w,h,_)=img_filtering.shape
    w_line=w//cell_size
    h_line=h//cell_size
    for i in range(w_line):
        img_filtering[i*cell_size,:,]=60
    for j in range(h_line):
        img_filtering[:,j*cell_size,]=60
    a=[]
    b=[]
    def on_EVENT_LBUTTONDOWN(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            xy="%d,%d"%(x,y)
            a.append(x)
            b.append(y)
            cv2.circle(img_filtering,(x,y),1,(255,0,0),thickness=-1)
            cv2.putText(img_filtering,xy,((x,y)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),thickness=2)
            cv2.imshow("image",img_filtering)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",on_EVENT_LBUTTONDOWN)
    cv2.imshow("image",img_filtering)
    cv2.waitKey(0)

    c=pow(pow(a[1]-a[0],2)+pow(b[1]-b[0],2),0.5)
    img1=cv2.line(img_filtering,(a[0],b[0]),(a[1],b[1]),color=(0,0,255),thickness=2)

    ratio=c/PD_pixel_distance
    d=value*ratio

    img1=cv2.putText(img_filtering,f"head={'%.2f'%d}",(int((a[0]+a[1])/2+10),int((b[0]+b[1])/2)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),thickness=2)

    cv2.imshow('result',img1)
    cv2.waitKey()

    print(f"头高：head={'%.2f'%d} cm")

    Pl=1.147*d-17.136
    rand=random.uniform(3,5)
    Cl=Pl-rand
    print(f"生理性推荐侧卧枕高Pl为：{'%.1f'%Pl} cm")
    print(f"生理性推荐仰卧枕高Cl为：{'%.1f'%Cl} cm")
    Pl=(f"{'%.1f' %round(Pl,1)} cm")
    Cl=(f"{'%.1f' %round(Cl,1)} cm")
    # P.append(Pl)
    # C.append(Cl)
#    result={Pl,Cl}
    return Pl,Cl