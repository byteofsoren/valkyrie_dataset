# import cv2 as cv
import cv2
import numpy as np
import os
import glob
# import argparse
# parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
# parser.add_argument('--input1', help='Path to input image 1.', default='box.png')
# parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')
# args = parser.parse_args()
# img1 = cv2.imread(cv2.samples.findFile(args.input1), cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(cv2.samples.findFile(args.input2), cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("./data/human_straight_onback/img_020.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("./data/human4_onside/img_014.png", cv2.IMREAD_GRAYSCALE)

def matcher(img,board):
    img1 = board
    img2 = img

    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    # detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    # detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    # detector = cv2.xfeatures2d.SURF_create(hessianThreshold=minHessian)
    # detector = cv2.xfeatures2d.SURF_create()
    detector = cv2.xfeatures2d.SIFT_create()
    # detector = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

def matcher_orb(img,board):
    img1 = board
    img2 = img

    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Brute force matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    matches = bf.match(descriptors1,descriptors2)
    matches = sorted(matches,key=lambda x:x.distance)

    matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2,matches[:50], None, flags=2)
    return matching_result

if __name__ == "__main__":
    bod = cv2.imread("./checker_board.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("./data/human4_onside/img_014.png", cv2.IMREAD_GRAYSCALE)
    sel = dict()
    index=1

    for directory in glob.glob("data/*"):
        print(f"{index} {directory}")
        sel[index] = directory
        index += 1
    ch = int(input(f"select image 0 to {index}: "))
    # print(sel)
    seldir = sel[ch]
    imgid = [images  for images in glob.glob(f"{seldir}/*.png")]
    # print(imgid)
    resultdir = seldir.replace("data/","results/")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    else:
        print("dir existed")

    for img in imgid:
        imgname = img.replace(f"{seldir}/","")
        imgdata = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_matches = matcher(imgdata,bod)
        # img_matches = matcher_orb(imgdata,bod)
        # write_path
        cv2.imwrite(f"{resultdir}/{imgname}", img_matches)
        # -- Show detected matches
        # cv2.imshow('Good Matches', img_matches)

        # cv2.imwrite("/home/ros/repos/valkyrie_dataset/temp/boad+image.png", img_matches)
        # cv2.waitKey()
