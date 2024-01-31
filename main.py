import cv2
import numpy as np

def detect_garbage(reference_image_path):
    # Load the reference image
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    # Create a feature detector (ORB is used here, you can experiment with other methods)
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(reference_image, None)

    # Open the local camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while cap.isOpened():
        # Read frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the keypoints and descriptors with ORB in the live stream frame
        kp2, des2 = orb.detectAndCompute(gray_frame, None)

        # Use the BFMatcher to find the best matches between descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # If enough good matches are found, consider it a match
        if len(good_matches) > 20:
            print("Garbage detected!")

        # Display the frame
        cv2.imshow("Live Stream", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows on exit
    cap.release()
    cv2.destroyAllWindows()

# Example usage
reference_image_path = "garbage_reference.png"
detect_garbage(reference_image_path)