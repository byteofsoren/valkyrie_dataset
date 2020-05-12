import cv2
import pandas
from time import sleep

cv2.namedWindow("preview")
vc = cv2.VideoCapture(2)
data = pandas.read_csv("poses.csv")

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

print(data)
cam = {100:'left',10:'center',1:'right'}

# breakpoint()
if rval:
    for index, row in data.iterrows():
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        # print(f'key={key}')
        angle=row['angle']
        dist=row['dist']
        camera=row['camera']
        height=row['height']
        print("aoeu")
        filename = f'{angle:03d}_{height}_{dist:02d}_{camera:03d}.png'
        print(f'{filename}')
        print(f'Angle={angle}, height={height}, distance={dist}, camera pos = {cam[camera]} ')
        while key != 32:
            sleep(0.1)
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            # print(key)
            if key == 27:
                exit()

        cv2.imwrite(f'data/{filename}', frame)
cv2.destroyWindow("preview")
