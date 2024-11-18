import cv2
import numpy as np
from matplotlib import pyplot as plt

def sift_match(img1, img2):
    # plt.imshow(img1)
    # plt.show()
    # print("Min img1:", np.min(img1))
    # print("Max img1:", np.max(img1))
    # img1 = np.uint8(img1 * 255) 
    # img2 = np.uint8(img2 * 255) 
    # img1 = img1.astype('uint8')
    # img2 = img2.astype('uint8')
    # print("IMG1")
    # print(img1)
    # print("====")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # print("Min value:", np.min(gray1))
    # print("Max value:", np.max(gray1))
    # plt.imshow(gray1, cmap='gray')
    # plt.show()

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, ds1 = sift.detectAndCompute(gray1, None)
    kp2, ds2 = sift.detectAndCompute(gray2, None)

    # print("DS: \n")
    # print(ds1)
    # print("===")

    # Match descriptors using FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # Using KD-tree algorithm
    search_params = dict(checks=50)  # Number of times the tree is traversed
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(ds1, ds2, k=2)
    # print(matches)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract the matched keypoints' positions
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    print(f'  Keypoint detection and matching done ({len(good_matches)} matches found).')
    print(f'FROM SIFT: {pts1.shape}')
    return pts1, pts2

# Example usage:
# img1 = cv2.imread('path/to/image1.jpg')
# img2 = cv2.imread('path/to/image2.jpg')
# pts1, pts2 = sift_match(img1, img2)
