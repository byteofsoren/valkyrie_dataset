# valkyrie dataset
This dataset is used in the Valkyrie system at UMA.

Directory structure
```|
   |-- data/        # contains the datasets
   |-- poses.csv    # Each pose that should be taken
   |-- take_img.py  # The program that takes the images stated in pose.csv
   |-- READ.md      # This file
```

## How to take images
To create a new data set of images just start the `take_img.py` like this.
```
$ python3 take_imag.py
```
if nothing happens then perhaps the the camera settings insidet he `take_img.py` need to change.

The row that creates the `vc` objects linke this:
```
cv2.namedWindow("preview")
vc = cv2.VideoCapture(2)   # <-- change this row
data = pandas.read_csv("poses.csv")
```
Needs to refflect what camera  number your camera have in `/dev/video*`
