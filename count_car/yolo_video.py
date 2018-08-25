import sys

# if len(sys.argv) < 2:
#     print("Usage: $ python {0} [video_path] [output_path(optional)]", sys.argv[0])
#     exit()

from yolo import YOLO
from yolo import detect_video

def main():
    # video_path = sys.argv[1]
    # if len(sys.argv) > 2:
    #     output_path = sys.argv[2]
    #     detect_video(YOLO(), video_path, output_path)
    # else:
    #     detect_video(YOLO(), video_path)
    video_path='./data/video/44.mp4'
    name = video_path.split('video/')[1].split('.mp4')
    # print(name[0])
    output_path='/home/tsl/keras-yolo3/data/result/'+name[0]+'.mp4'
    detect_video(YOLO(),video_path,output_path)

if __name__ == '__main__':
    main()

