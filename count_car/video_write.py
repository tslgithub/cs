import sys

# if len(sys.argv) < 2:
#     print("Usage: $ python {0} [video_path] [output_path(optional)]", sys.argv[0])
#     exit()

from yolo import YOLO
# from yolo import detect_video
import cv2
import tools

def detect_video( video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1920,1080))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break



def main():
    # video_path = sys.argv[1]
    # if len(sys.argv) > 2:
    #     output_path = sys.argv[2]
    #     detect_video(YOLO(), video_path, output_path)
    # else:
    #     detect_video(YOLO(), video_path)
    video_path='./data/video/40.mp4'
    output_path='/home/tsl/keras-yolo3/result.mp4'
    detect_video(video_path,output_path)

if __name__ == '__main__':
    main()

