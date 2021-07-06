import cv2
import numpy as np


class EMA:
    '''
        input     : tuple()
        operation : list()
        output    : tuple()
    '''
    def __init__(self, points=[], alpha=0.4, margin=150):
        self.points = points
        self.alpha = alpha
        self.margin = margin

    def calc(self, points):
        newpoints = [] #volatile

        if self.points == []: # initial condition
            self.points = list(points)

        for i in range(len(points)):
            if abs(self.points[i] - points[i])>= self.margin: # long distance condition
                self.points[i] = int(0.7*points[i] + self.points[i]*0.3)

            one = int(self.points[i]*(1-self.alpha) + self.alpha*points[i])
            newpoints.append(one)

        self.points = newpoints.copy()
        return tuple(newpoints)

def draw_angle(image, landmark, p1, p2, scale):
    #- nose
    nose = (int(landmark[4] * scale[0]), int(landmark[5] * scale[1]))

    ps1 = (int(p1[0] * scale[0]), int(p1[1] * scale[1]))
    ps2 = (int(p2[0] * scale[0]), int(p2[1] * scale[1]))

    print(nose)
    print(ps1)

    #- draw nose
    cv2.circle(image, nose, 5, (255,0,0))
    #- draw angle
    cv2.line(image, ps1, ps2, (0, 0, 255), 3)


def get_camera_matrix(image):
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_metrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype = "float")
    return camera_metrix

def get_model_points():
    # good to rnet
    model_points = np.array([   (   0.0,    0.0,    0.0),   # Nose tip
                                (   0.0, -330.0,  -65.0),   # Chin
                                (-160.0,  180.0, -160.0),   # Left eye left corner
                                ( 160.0,  180.0, -160.0),   # Right eye right corne
                                (-150.0, -150.0, -150.0),   # Left Mouth corner
                                ( 150.0, -150.0, -150.0)])  # Right mouth corner
    return model_points

def get_default_image_points():
    image_points = np.array([(0, 0),     # Nose tip
                             (0, 0),     # chin
                             (0, 0),     # Left eye left corner
                             (0, 0),     # Right eye right corne
                             (0, 0),     # Left Mouth corner
                             (0, 0)      # Right mouth corner
                            ], dtype="float")
    return image_points

def get_biggest_face(boxes_c, landmarks):
    '''
        - inputs 
            n is the number of candidates
            boxes_c  : 2d-array [[x1, y1, x2, y2, score]] - (n, 5)
            landmark : 2d-array [[leye-x, leye-y, reye-x, reye-y, ... ]] (n, 12)
        - outputs
            biggest bbox_c, landmark
    '''

    get_areas = (boxes_c[:,2] - boxes_c[:,0]) * (boxes_c[:,3] - boxes_c[:,1])
    bidx   = np.argmax(get_areas, axis=0)

    bbox_c  = np.expand_dims(boxes_c[bidx, :],   axis=0)
    lmk     = np.expand_dims(landmarks[bidx, :], axis=0)

    return bbox_c, lmk




 
