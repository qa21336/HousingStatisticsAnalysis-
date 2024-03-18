import numpy as np
import cv2

def calculate_distance(disparity, focal_length, baseline):
    """
    Calculate distance to an object based on its disparity, focal length, and baseline.
    :param disparity: Disparity map
    :param focal_length: Focal length of the camera
    :param baseline: Distance between the two camera positions
    :return: Depth map (distance to the objects)
    """
    depth_map = focal_length * baseline / disparity
    return depth_map

def main():
    # Load stereo images
    img_left = cv2.imread('im0.png')
    img_right = cv2.imread('im1.png')

    # Convert images to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # StereoSGBM parameter setup
    min_disparity = 55
    num_disparities = 320
    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size
    )

    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Set focal length (in pixels) and baseline (in meters)
    focal_length = 748.4
    baseline = 0.53662 #0.1  in m # 10 cm

    # Calculate depth map (distance to objects)
    depth_map = calculate_distance(disparity, focal_length, baseline)

    # Display disparity and depth map
    cv2.imshow('Disparity', disparity / num_disparities)
    cv2.imshow('Depth Map', depth_map / 100)  # Convert to meters for better visualization
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
