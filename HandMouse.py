
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:00:37 2021

------ HAND TRACKER -------
@author: 
"""
import cv2 as cv
import mediapipe as mp
import math
import numpy as np
import autopy 


def get_face_landmarks(image, landmark, face_output):
    left, top, right, bottom  =999, 999, -1, -1
    
    height, width =image.shape[:2]
    
    is_valid_face = False
    for face in landmark:
        point = face_output.multi_face_landmarks[0].landmark[face]
        ptX, ptY = int(point.x * width), int(point.y * height)
        
        if left>ptX :
            left = ptX
            
        if right<ptX :
            right = ptX
            
        if top > ptY:
            top = ptY
            
        if bottom < ptY:
            bottom = ptY
            
        is_valid_face = True

        
    return is_valid_face, left, top, right, bottom



def get_landmark_regions(image, face_output, cam_width, cam_height, selected_obj=0):
    
    if selected_obj == LIPS:
        is_valid_face,left,top,right, bottom = get_face_landmarks(image, LIPS_MESH, face_output)
        
    if selected_obj == EYES:    
        is_valid_face_l,left_l,top_l,right_l, bottom_l = get_face_landmarks(image, LEFT_EYE_MESH, face_output)
        is_valid_face_r,left_r,top_r,right_r, bottom_r = get_face_landmarks(image, RIGHT_EYE_MESH, face_output)
        
        if is_valid_face_l and is_valid_face_r:
            is_valid_face = True
            
        offset =  left_l - right_r
        offset_reduce = int(0.5 *offset)
        
        top = np.amin([top_l,top_r])
        bottom = np.amax([bottom_l,bottom_r]) 
        
        top -=offset_reduce
        bottom+=offset_reduce
        
        left, right  = left_r - offset_reduce, right_l+ offset_reduce, 
        
    if selected_obj == FACE:    
        is_valid_face,left,top,right, bottom = get_face_landmarks(image, FACE_MESH, face_output)
                              
        offset_side =  int(0.1*(bottom - top))
        height = int(0.25 * (bottom - top))
        
        bottom = top + height
        top = top -  int(2.0 *height)
        left -=offset_side
        right +=offset_side
        
    if selected_obj == NOSE:
        
        is_valid_face_n,left_n,top_n,right_n, bottom_n = get_face_landmarks(image, LOWER_NOSE_MESH, face_output)
        is_valid_face_l,left_l,top_l,right_l, bottom_l = get_face_landmarks(image, LIPS_MESH, face_output)
        
        height = top_l - bottom_n
        left, right = left_n, right_n
        top, bottom = bottom_n, top_l + int(0.5*height)
        
        if is_valid_face_n and is_valid_face_l:
            is_valid_face = True
            
    if (left < 0 or top < 0 or right< 0 or bottom < 0)  or (left > cam_width or top > cam_height or right> cam_width or bottom > cam_height) :
            is_valid_face = False
            
    return is_valid_face, left, top, right, bottom



def select_object(pt_index_tip, pt_index_mcp, 
                  pt_middle_tip, pt_middle_mcp, 
                  object_focus, selected_obj):   
          
    distance_index = math.dist(pt_index_tip, pt_index_mcp)
    distance_middle_index = math.dist(pt_index_tip, pt_middle_tip)
    distance_middle = math.dist(pt_middle_tip, pt_middle_mcp)           
    
    if distance_middle_index > 0 and distance_middle >0:
        ratio_distance = distance_middle_index/distance_index
        
        ratio_middle = distance_middle_index/distance_middle
        
        #right click
        if ratio_middle >0.8 and distance_middle < distance_index :
            if object_focus !=-1:
                selected_obj = -1
        #left click    
        if ratio_distance > 0.8 and distance_index < 0.7*distance_middle:
            
            if object_focus !=-1:
                selected_obj = object_focus
                
    return selected_obj;          

def draw_focus_obj(pt_index_tip, object_focus):
    
    y_pos = -1
    seleted = -1
    if pt_index_tip[0] > 0 and pt_index_tip[1] > 0 and pt_index_tip[0] < width_obj and  pt_index_tip[1] < height_obj:
                
        if pt_index_tip[1] < object_ind_height:
          seleted = 3
          y_pos = 0
          
        if pt_index_tip[1] > object_ind_height and pt_index_tip[1] <  2 * object_ind_height:
          seleted =2   
          y_pos = object_ind_height
        if pt_index_tip[1] > 2 * object_ind_height and pt_index_tip[1] <  3 * object_ind_height:
          seleted =1
          y_pos = 2 * object_ind_height
          
        if pt_index_tip[1] > 3 * object_ind_height and pt_index_tip[1] <  4 * object_ind_height:
          seleted =0
          y_pos = 3 * object_ind_height
        
        if y_pos != -1 and seleted !=-1:
            cv.rectangle(image_org, (0, y_pos), (width_obj, y_pos+object_ind_height), (0,255,0), 4)
        
            object_focus = seleted
        
    return object_focus


def get_index_middle_finger_positions(image, output):
    
    height, width = image.shape[:2]
                       
    handlandmark = output.multi_hand_landmarks[0]
    
    for index, hand in enumerate(handlandmark.landmark):
                       
        if index == INDEX_FINGER_MCP:
            x,y = int(hand.x * width), int(hand.y * height)
            
            pt_index_mcp = (x, y)
            
        if index == INDEX_FINGER_TIP:
            x,y = int(hand.x * width), int(hand.y * height)
            
            pt_index_tip = (x, y)
            
        if index == MIDDLE_FINGER_MCP:
            x,y = int(hand.x * width), int(hand.y * height)
            
            pt_middle_mcp = (x, y)
            
        if index == MIDDLE_FINGER_TIP:
            x,y = int(hand.x * width), int(hand.y * height)
            
            pt_middle_tip = (x, y)
            
    return pt_index_tip, pt_index_mcp, pt_middle_tip, pt_middle_mcp


def get_masked_object(image_obj, cam_height):
    
    height_, width_, _ = image_obj.shape
    
    scale_ratio = cam_height/height_
    
    image_resize = cv.resize(image_obj, ( int(width_*scale_ratio) , int(height_*scale_ratio)))
    
    height_obj, width_obj, _ = image_resize.shape
    
    
    image_resize_gray = cv.cvtColor(image_resize, cv.COLOR_BGR2GRAY)
    
    
    res, image_gray = cv.threshold(image_resize_gray, 200, 255, cv.THRESH_BINARY_INV)
    
    erosion = np.ones((7,7), np.uint8)
    image_gray = cv.erode(image_gray, erosion)
    
    #Fill up the images for holes
    h, w = image_gray.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    image_floodfill = image_gray.copy()
    cv.floodFill(image_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv.bitwise_not(image_floodfill)
    
    
    img_out = image_gray | im_floodfill_inv
    
    mask_inv_all = cv.bitwise_not(img_out)
    
    return image_resize, img_out, mask_inv_all



hand_object = mp.solutions.hands

face_mesh = mp.solutions.face_mesh

draw_utils = mp.solutions.drawing_utils

drwlandmarks = draw_utils.DrawingSpec( (255,0, 0), thickness= 2, circle_radius=3)
drwconnections= draw_utils.DrawingSpec( (0, 0, 255), thickness= 2)


STATIC_IMAGE = False
MAX_NO_OF_HANDS = 2
DETECT_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

INDEX_FINGER_MCP = hand_object.HandLandmark.INDEX_FINGER_MCP
INDEX_FINGER_TIP = hand_object.HandLandmark.INDEX_FINGER_TIP
MIDDLE_FINGER_TIP = hand_object.HandLandmark.MIDDLE_FINGER_TIP
MIDDLE_FINGER_MCP = hand_object.HandLandmark.MIDDLE_FINGER_MCP


LIPS_MESH=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
       185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

FACE_MESH=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]


LEFT_EYE_MESH = [263, 249, 390, 373, 374, 380, 381, 382, 362,263, 466, 388, 387, 386, 385, 384, 398, 362]

RIGHT_EYE_MESH = [33, 7, 163 ,144, 145, 153, 154, 155, 133,33, 246, 161, 160, 159, 158, 157, 173, 133]

LOWER_NOSE_MESH = [205, 203, 98, 97, 2, 327, 423, 425]

LIPS = 0
FACE = 1
NOSE = 3
EYES = 2

hand_model = hand_object.Hands(static_image_mode = STATIC_IMAGE,
                               max_num_hands = MAX_NO_OF_HANDS,
                               min_detection_confidence = DETECT_CONFIDENCE,
                               min_tracking_confidence=TRACKING_CONFIDENCE)

face_model = face_mesh.FaceMesh(static_image_mode = STATIC_IMAGE,
                                max_num_faces = 1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


path = 'C:/MEDIA_PIPE/VID-20211028-WA0002.mp4'

capture = cv.VideoCapture(0)


screen_width, screen_height = autopy.screen.size()
cam_width = 1280
cam_height = 720

capture.set(3, cam_width)
capture.set(4, cam_height)

load_object_path = "C:/MEDIA_PIPE/FaceObjects.png"
image_obj = cv.imread(load_object_path) 


image_resize, img_out, mask_inv_all = get_masked_object(image_obj, cam_height)
height_obj, width_obj, _ = image_resize.shape

#-- ends fill up ---


valid_contours = []
locations =[]
contours, heirarcy = cv.findContours(img_out, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in contours:
    
    if cv.contourArea(contour) > 30 :
        valid_contours.append(contour)       
        print(cv.boundingRect(contour))
        locations.append(cv.boundingRect(contour))

valid_contours = np.array(valid_contours)


object_ind_height = int(height_obj/4)

selected_obj = -1

while True:
    
    result, image_org = capture.read()
    
    if result:
        
        image = cv.cvtColor(image_org, cv.COLOR_BGR2RGB)
        
        
        object_focus = -1
        output = hand_model.process(image)
        
        if output.multi_hand_landmarks:
            
            pt_index_tip,pt_index_mcp, pt_middle_tip,  pt_middle_mcp = get_index_middle_finger_positions(image, output)
                          
            cv.circle(image_org, pt_index_tip, 5, (50,200, 50), 8)
            cv.circle(image_org, pt_middle_tip, 5, (50,50, 200), 8)
            
                       
            image_org[0:height_obj, 0:width_obj] = image_resize
            #autopy.mouse.move(int(pt_middle_tip_window_x), int(pt_middle_tip_window_y))
            
                      
            object_focus = draw_focus_obj(pt_index_tip, object_focus)
            
                
            #print(object_focus)
                       
            selected_obj = select_object(pt_index_tip, pt_index_mcp, 
                                         pt_middle_tip, pt_middle_mcp, 
                                         object_focus, selected_obj)
            
            if selected_obj !=-1:
                cv.drawContours(image=image_org, contours= valid_contours[selected_obj], contourIdx=-1, color=(0,255,0), thickness=5)
                    
            for hand in output.multi_hand_landmarks:
                draw_utils.draw_landmarks(image_org,
                                              hand,
                                              hand_object.HAND_CONNECTIONS,
                                              landmark_drawing_spec=drwlandmarks,
                                              connection_drawing_spec=drwconnections)
                
              
        if selected_obj != -1:
            
                #.......Get Face Locations ....
                                         
                face_output = face_model.process(image)
                
                is_valid_face = False
                if face_output.multi_face_landmarks:
                    is_valid_face, left, top, right, bottom = get_landmark_regions(image, face_output, cam_width, cam_height, selected_obj)
                    print(is_valid_face, left, top, right, bottom)
                                               
                if is_valid_face:                                       
                    print(left, top, right, bottom)
                    image_region = image_org[top:bottom, left:right]
                    
                    crop_height, crop_width = image_region.shape[:2]
                    
                    [x,y,w,h] = locations[selected_obj]
                    
                    object_crop =  image_resize[y:y+h, x:x+w]
                    mask_obj = img_out[y:y+h, x:x+w]
                    mask_inv = mask_inv_all[y:y+h, x:x+w]
                    
                    
                    object_crop = cv.resize(object_crop, (crop_width, crop_height))
                    mask_obj = cv.resize(mask_obj, (crop_width, crop_height))
                    mask_inv = cv.resize(mask_inv, (crop_width, crop_height))
                
                    
                    image_for = cv.bitwise_and(object_crop, object_crop, mask=mask_obj)
                    image_back = cv.bitwise_and(image_region, image_region, mask=mask_inv)                   
                    image_out = cv.add(image_back, image_for)

                    image_org[top:bottom, left:right] = image_out
                    
          
        image_org = cv.flip(image_org, 1)
        cv.imshow("HAND", image_org)
        if cv.waitKey(30) & 255 == 27:
            break
        
capture.release()
cv.destroyAllWindows()
                
                
                
                
