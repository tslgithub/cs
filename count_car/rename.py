import os

root = os.getcwd()
video_path = os.path.join(root,'data/video/')
i=10
for video in os.listdir(video_path):
    src = os.path.join(video_path,video)
    dst = os.path.join(video_path,str(i)+'.mp4')
    os.rename(src,dst)
    i+=1