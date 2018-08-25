from moviepy.editor import *
import os
import cv2
import numpy as np
import tools
L=[]



def video_msg(video):
    vid = cv2.VideoCapture(video)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    ret, frame = vid.read()
    video_fps_list = []
    while ret:
        ret, frame = vid.read()
        video_fps_list.append([frame])

    return video_fps_list,video_fps,video_size

def main():
    video_root = os.path.join(os.getcwd(), 'data/result/')
    video1 = os.path.join(video_root, '44.mp4')
    video2 = os.path.join(video_root, '44_1.mp4')
    name = '44'
    result = os.path.join(video_root,name+'_result.mp4')
    video_fps_list1, video_fps, video_size  = video_msg(video2)
    video_fps_list2, video_fps, video_size  = video_msg(video2)

    cap = cv2.VideoCapture(video1)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(result, fourcc, 30.0, (1920,1080))
    for i in range(len(video_fps_list1)-1):
        frame = np.concatenate([video_fps_list1[i][0],video_fps_list2[i][0]],axis=1)

        out.write(frame)
        cv2.imshow('frame', frame)




if __name__ == '__main__':
    main()
