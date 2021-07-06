import numpy as np
import cv2

def head_gaze_estimation(image, landmark, camera_matrix, model_points, image_points):
    #- Assuming no lens distortion
    dist_coeffs = np.zeros((4,1)) 

    image_points = np.array([(landmark[0:, 4], landmark[0:, 5]),  # nose
                             (landmark[0:,10], landmark[0:,11]),  # chin
                             (landmark[0:, 0], landmark[0:, 1]),  # l-eye
                             (landmark[0:, 2], landmark[0:, 3]),  # r-eye
                             (landmark[0:, 6], landmark[0:, 7]),  # l-mouse
                             (landmark[0:, 8], landmark[0:, 9])], # r-mouse
                             dtype="float")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))

    euler_angles_radians = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    
    pitch, yaw, roll = [theta for theta in euler_angles_radians]

    if pitch > 0:
        pitch = 180 - pitch
    elif pitch < 0:
        pitch = -180 - pitch

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = np.asarray( (int(image_points[0][0]),        int(image_points[0][1])) )
    p2 = np.asarray( (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])) )
    
    return p1, p2, roll, yaw, pitch
