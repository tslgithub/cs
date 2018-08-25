ffmpeg \
  -i /home/tsl/keras-yolo3/data/result/44.mp4 \
  -i /home/tsl/keras-yolo3/data/result/44_1.mp4 \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  output.mp4

