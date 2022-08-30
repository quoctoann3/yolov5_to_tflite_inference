from yolov5_tflite_inference import yolov5_tflite
import argparse
import cv2
from PIL import Image
from utils import letterbox_image, scale_coords
import numpy as np
from time import time
from datetime import datetime
def detect_video(weights,webcam,img_size,conf_thres,iou_thres):

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoCapture(webcam)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    h = 320
    w = 320
    print(w,h)
    #h = 1280
    #w = 720
    #result_video_filepath =  'webcam_yolov5_output.mp4'
    #out  = cv2.VideoWriter(result_video_filepath,fourcc,int(fps),(h,w))

    yolov5_tflite_obj = yolov5_tflite(weights,img_size,conf_thres,iou_thres)

    size = (img_size,img_size)
    no_of_frames = 0
    # Get input index]

    fps_sleep = int(1000 / fps)

    print('* Capture width:', w)
    print('* Capture height:', h)
    print('* Capture FPS:', fps)
    print('{}{}'.format('model loaded: ','model.tflite'))
    print('model running...')
    last_time = datetime.now()
    frames = 0
    while True:
        frames += 1
        check, frame = video.read()
        # compute fps: current_time - last_time
        delta_time = datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        #cur_fps = np.around(frames / elapsed_time, 1)
        cur_fps = np.int(frames / elapsed_time)
        cv2.putText(frame,str(datetime.now().replace(microsecond=0)),(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(48,41,33),2,cv2.LINE_AA)
        cv2.putText(frame, 'FPS: ' + str(cur_fps), (550, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (48,41,33), 2, cv2.LINE_AA)
        if not check:
            break
        #frame = cv2.resize(frame,(h,w))
        #no_of_frames += 1
        image_resized = letterbox_image(Image.fromarray(frame),size)
        image_array = np.asarray(image_resized)

        normalized_image_array = image_array.astype(np.float32) / 255.0
        result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(normalized_image_array)
        if len(result_boxes) > 0:
            result_boxes = scale_coords(size,np.array(result_boxes),(w,h))
                                                        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        # org 
        org = (20, 40) 
            
        # fontScale 
        fontScale = 0.5
            
        # Line thickness of 1 px 
        thickness = 1

        for i,r in enumerate(result_boxes):

            org = (int(r[0]),int(r[1]))
            cv2.rectangle(frame, (int(r[0]),int(r[1])), (int(r[2]),int(r[3])), (0, 43, 255), 2)
            cv2.putText(frame, str(int(100*result_scores[i])) + '%  ' + str(result_class_names[i]), org, font,  
                        fontScale, (0, 43, 255), thickness, cv2.LINE_AA)
            
            cv2.imwrite('./defect_detected/'+str((datetime.now().replace(microsecond=0)))+'.jpg',frame)
            start = time()
            inf_time = time() - start
            print('Detected: conf:{}%, infe_time:{}s ___{}'.format(str(round(result_scores[i],2)),str(round(inf_time*1000,3)),str(datetime.now().replace(microsecond=0))))
        #out.write(frame)

        #uncomment below lines to see the output
        cv2.imshow('test',frame)   

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('stoped!')
            break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights', type=str, default='model.tflite', help='model.tflite path(s)')
    parser.add_argument('-wc','--webcam', type=int, default=0, help='webcam number 0,1,2 etc.') 
    parser.add_argument('--img_size', type=int, default=320, help='image size')  
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='IOU threshold for NMS')

    
    opt = parser.parse_args()
    
    print(opt)
    detect_video(opt.weights,opt.webcam,opt.img_size,opt.conf_thres,opt.iou_thres)
