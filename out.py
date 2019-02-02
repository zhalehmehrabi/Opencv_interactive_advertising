import numpy as np
import cv2 as cv
import random
import time


def show(offx, offy, dst, image):
    for xim in range(widthball):
        for yim in range(heightball):
            if image[yim, xim, 0] != 0:
                dst[offy + yim, offx + xim] = image[yim, xim]


if __name__ == "__main__":
    baseCascadePath = "D:\Programming language\opencv\opencv\sources\data\haarcascades"

    faceCascadeFilePath = baseCascadePath + "haarcascade_frontalface_default.xml"
    noseCascadeFilePath = baseCascadePath + "haarcascade_mcs_nose.xml"

    faceCascade = cv.CascadeClassifier(faceCascadeFilePath)
    noseCascade = cv.CascadeClassifier(noseCascadeFilePath)

    faceCascade.load(
        "D:\Programming language\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
    noseCascade.load("D:\Programming language\opencv\opencv\sources\data\haarcascades\haarcascade_mcs_nose.xml")

    imgHat = cv.imread('happy.png', -1)

    image1 = cv.imread('image1.png', -1)
    image1 = image1[:, :, 0:3]
    x1 = random.randint(50, 150)
    y1 = random.randint(50, 100)
    xm1 = x1 + 25
    ym1 = y1 + 50

    image2 = cv.imread('image2.png', -1)
    image2 = image2[:, :, 0:3]

    x2 = random.randint(150, 300)
    y2 = random.randint(50, 100)
    xm2 = x2 + 25
    ym2 = y2 + 50

    image3 = cv.imread('image3.png', -1)
    image3 = image3[:, :, 0:3]

    x3 = random.randint(300, 450)
    y3 = random.randint(50, 100)

    xm3 = x3 + 25
    ym3 = y3 + 50

    heightball = 100
    widthball = 50

    image4 = cv.imread('image4.png', -1)
    image4 = image4[:, :, 0:3]
    image5 = cv.imread('image5.png', -1)
    image5 = image5[:, :, 0:3]
    image6 = cv.imread('image6.png', -1)
    image6 = image6[:, :, 0:3]
    image7 = cv.imread('image7.png', -1)
    image7 = image7[:, :, 0:3]

    image1 = cv.resize(image1, (widthball, heightball), interpolation=cv.INTER_AREA)
    image2 = cv.resize(image2, (widthball, heightball), interpolation=cv.INTER_AREA)
    image3 = cv.resize(image3, (widthball, heightball), interpolation=cv.INTER_AREA)
    image4 = cv.resize(image4, (widthball, heightball), interpolation=cv.INTER_AREA)
    image5 = cv.resize(image5, (widthball, heightball), interpolation=cv.INTER_AREA)
    image6 = cv.resize(image6, (widthball, heightball), interpolation=cv.INTER_AREA)
    image7 = cv.resize(image7, (widthball, heightball), interpolation=cv.INTER_AREA)

    i = 0
    j1 = 0
    j2 = 0
    j3 = 0
    imgHat = imgHat[:, :, 0:3]
    origHatHeight, origHatWidth = imgHat.shape[:2]
    cap = cv.VideoCapture('output.avi')
    fgbg = cv.createBackgroundSubtractorMOG2()

    controller1 = 1
    controller2 = 1
    controller3 = 1
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('edited.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        i += 1
        if controller1 == 1:
            if i % 3 == 0:
                asli1 = image1
            elif i % 3 == 1:
                asli1 = image2
            elif i % 3 == 2:
                asli1 = image3
        else:
            if j1 % 4 == 0:
                asli1 = image4
            elif j1 % 4 == 1:
                asli1 = image5
            elif j1 % 4 == 2:
                asli1 = image6
            elif j1 % 4 == 3:
                asli1 = image7
                x1 = random.randint(50, 150)
                y1 = random.randint(50, 100)
                xm1 = x1 + 25
                ym1 = y1 + 50
                controller1 = 1
            j1 += 1

        if controller2 == 1:
            if i % 3 == 1:
                asli2 = image1
            elif i % 3 == 2:
                asli2 = image2
            elif i % 3 == 0:
                asli2 = image3
        else:
            if j2 % 4 == 0:
                asli2 = image4
            elif j2 % 4 == 1:
                asli2 = image5
            elif j2 % 4 == 2:
                asli2 = image6
            elif j2 % 4 == 3:
                asli2 = image7
                x2 = random.randint(150, 300)
                y2 = random.randint(50, 100)
                xm2 = x2 + 25
                ym2 = y2 + 50
                controller2 = 1
            j2 += 1
        if controller3 == 1:
            if i % 3 == 2:
                asli3 = image1
            elif i % 3 == 0:
                asli3 = image2
            elif i % 3 == 1:
                asli3 = image3
        else:
            if j3 % 4 == 0:
                asli3 = image4
            elif j3 % 4 == 1:
                asli3 = image5
            elif j3 % 4 == 2:
                asli3 = image6
            elif j3 % 4 == 3:
                asli3 = image7
                x3 = random.randint(300, 450)
                y3 = random.randint(50, 100)

                xm3 = x3 + 25
                ym3 = y3 + 50
                controller3 = 1
            j3 += 1
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.bilateralFilter(fgmask, 9, 75, 75)

        if fgmask[ym1, xm1] > 130:
            controller1 = 0
        if fgmask[ym2, xm2] > 130:
            controller2 = 0
        if fgmask[ym3, xm3] > 130:
            controller3 = 0

        # ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # _, contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # cv.drawContours(frame, contours, 0, (0, 0, 255), 6)
        # cv.imshow('frame', fgmask)

        dst = frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv.CASCADE_SCALE_IMAGE
        )
        xkale = 0
        widthkale = 0
        for (x, y, w, h) in faces:
            xkale = x
            widthkale = w
            # face = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            hatWidth = w
            hatHeight = int((origHatHeight / origHatWidth) * hatWidth)
            hat = cv.resize(imgHat, (hatWidth, hatHeight), interpolation=cv.INTER_AREA)
            print(hat.shape)
            for x_hat in range(hatWidth):
                for y_hat in range(hatHeight):
                    if hat[y_hat, x_hat, 0] != 255:
                        dst[y + y_hat - h, x + x_hat] = hat[y_hat, x_hat]

            break# bara chan nfr
        show(x1, y1, dst, asli1)
        show(x2, y2, dst, asli2)
        show(x3, y3, dst, asli3)

        if ret == True:
            frame = cv.flip(frame, 0)
            # write the flipped frame
            rows, cols, channel = frame.shape
            M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 180, 1)
            dst = cv.warpAffine(frame, M, (cols, rows))
            out.write(dst)
        cv.imshow('frame', dst)
        pastframe = blurred
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
