#!/usr/bin/env python3
import cv2
import os
import logging
import numpy as np
import sys
import time
import math
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    
    return parser



def main():
    args = build_argparser().parse_args()
    previewFlags = args.previewFlags
    
    logger = logging.getLogger()
    inputFilePath = args.input
    inputFeeder = None
    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)
    
    modelPathDict = {'FaceDetectionModel':args.facedetectionmodel, 'FacialLandmarksDetectionModel':args.faciallandmarkmodel, 
    'GazeEstimationModel':args.gazeestimationmodel, 'HeadPoseEstimationModel':args.headposemodel}
    
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified "+fileNameKey+" xml file")
            exit(1)
            
    fdm = FaceDetectionModel(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(modelPathDict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    gem = GazeEstimationModel(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')
    
    inputFeeder.load_data()
    start = time.time()
    fdm.load_model()
    fdmload_time = time.time() - start
    load_time_message = "Loading time for Face Detection Model: {:.3f}ms".format(fdmload_time * 1000)
    print(load_time_message)
    start = time.time() 
    fldm.load_model()
    fldmload_time = time.time() - start
    load_time_message = "Loading time for Facial Landmark Model: {:.3f}ms".format(fldmload_time * 1000)
    print(load_time_message)
    start = time.time()
    hpem.load_model()
    hpemload_time = time.time() - start
    load_time_message = "Loading time for Head Pose Estimation Model: {:.3f}ms".format(hpemload_time * 1000)
    print(load_time_message)
    start=time.time()
    gem.load_model()
    gemload_time = time.time() - start
    load_time_message = "Loading time for Gaze Estimation Model: {:.3f}ms".format(gemload_time * 1000)
    print(load_time_message)
    total_load_time = gemload_time + fdmload_time + hpemload_time + fldmload_time
    load_time_message = "Loading time for all the Models: {:.3f}ms".format(total_load_time * 1000)
    print(load_time_message)

    frame_count = 0
    inf_start=time.time()
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        if type(croppedFace)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
        
        hp_out = hpem.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        total_inf_time = time.time() - inf_start
        inf_time_message = "Total Inference Time: {:.3f}s".format(total_inf_time)
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out) 

        if (not len(previewFlags)==0):
            preview_frame = frame.copy()
            if 'fd' in previewFlags:
                #cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3)
                preview_frame = croppedFace
            if 'fld' in previewFlags:
                cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                
            if 'hp' in previewFlags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            if 'ge' in previewFlags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                
            cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
                break
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    inputFeeder.close()
    print(inf_time_message)
    fps = frame_count / total_inf_time
    fps_message = "Total FPS: {:.3f} fps".format(fps)
    print(fps_message)

    

if __name__ == '__main__':
    main() 
 
