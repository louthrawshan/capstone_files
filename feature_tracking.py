import numpy as np
import cv2

# SOME USEFUL CODES:
# image1 = cv2.putText(image1, "Original", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# print(WIDTH//RESIZE_FACTOR*2, HEIGHT//RESIZE_FACTOR)

# read the video file
VIDEO_FILE = "../Dataset Videos/000242AA.MP4"
cap = cv2.VideoCapture(VIDEO_FILE)

# determine which part of the video to start from
START_FRAME = 3000
cap.set(1, START_FRAME)
ret, image = cap.read()

# resize the image
RESIZE_FACTOR = 3
HEIGHT, WIDTH = image.shape[0:2]
def resize_image(image):
        image = cv2.resize(image, (WIDTH//RESIZE_FACTOR, HEIGHT//RESIZE_FACTOR))
        return image

# undistort the image
CAMERA_MATRIX = np.array([[1230/RESIZE_FACTOR, 0, 960/RESIZE_FACTOR], [0, 1230/RESIZE_FACTOR, 540/RESIZE_FACTOR], [0, 0, 1]])
K = (-0.32, 0.126, 0, 0)
def undistort_image(image):
    image = cv2.undistort(image, CAMERA_MATRIX, K)
    return image

# create a fast detector
FAST_THRESHOLD = 40
NONMAX_SUPPRESSION = True
fast = cv2.FastFeatureDetector_create(FAST_THRESHOLD, NONMAX_SUPPRESSION)

# Display the image with keypoints
def draw_keypoints(image, kp):
    image_copy = image.copy()
    for point in kp:    
        pt = (int(point[0]), int(point[1]))
        cv2.rectangle(image_copy, (pt[0]-2, pt[1]-2), (pt[0]+2, pt[1]+2), (255, 0, 0), -1)
    return image_copy

# Normalize the color of the frame so that features can be more easily detected 
def equalize_one_layer(image_layer):
    return cv2.equalizeHist(image_layer)[..., np.newaxis]

def histogram_equalize(image):
    image = np.concatenate([equalize_one_layer(image[:, :, 0]), equalize_one_layer(image[:, :, 1]), equalize_one_layer(image[:, :, 2])], axis=2)
    return image

# creating an output video writer
out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter.fourcc(*'mp4v'), 60.0, (1280,360))

main_status = np.array([0])

min_points = MIN_POINTS = 200

# main loop
for i in range(10000):
    ret, current_image = cap.read()

    if not ret:
        break

    current_image = histogram_equalize(undistort_image(resize_image(current_image)))

    
    if main_status.shape[0]<min_points:
        ret, previous_image = cap.read()
        previous_image = histogram_equalize(undistort_image(resize_image(previous_image)))
        previous_keypoints = fast.detect(previous_image, None)
        previous_keypoints = np.float32([[point.pt[0], point.pt[1]] for point in previous_keypoints])

        main_status = np.arange(previous_keypoints.shape[0])
        main_kp = previous_keypoints
        main_index_correct = 0
        min_points = max(previous_keypoints.shape[0]/2, MIN_POINTS)
        restart = False

        image_show_1 = draw_keypoints(previous_image, previous_keypoints)

    # Track points. Then, remove points that were not tracked
    current_keypoints = np.zeros_like(previous_keypoints)
    if previous_keypoints != []:
        current_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(previous_image, current_image, previous_keypoints, current_keypoints)

    index_correct = 0
    for j in range(current_keypoints.shape[0]):
        pt = current_keypoints[j - index_correct]

        if (status[j] == 0 or pt[0]<0 or pt[1]<0):
            main_status = np.delete(main_status, j-index_correct, axis=0)
            current_keypoints = np.delete(current_keypoints, j-index_correct, axis=0)
            index_correct += 1

    previous_keypoints = current_keypoints
    previous_image = current_image

    # draw keypoints and then concatenate the original and current frames
    image_show_2 = draw_keypoints(current_image, current_keypoints)
    image_show = np.concatenate((image_show_1, image_show_2), axis=1)
    
    # show the frame
    cv2.imshow("keypoints", image_show)
    cv2.waitKey(1)
    
    # write to the output videop
    out.write(image_show)

out.release()


# When to re-start tracking:
#   1. After a few frames if too sparse
#   2. If all the points are clumped together (the area the points cover is too small, take 10 and 90th percentile and find the area they cover)