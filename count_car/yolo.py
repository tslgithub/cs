#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

import cv2
import tools

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from keras.utils import multi_gpu_model
gpu_num=1

# class YOLO(object):
class YOLO():
    # def __init__(self):
    def __init__(self,n='__'):
        self.n = "__"
        self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (320, 320) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()
        self.Number = 0
        self.Boxes = []
        self.line_down = 1000
        self.line_up = 950
        self.flag = 0
        # self.boxes=[]
        # self.car_number = 0

        # def __init__(self, n="__"):
        # _p 最后一次的Mat值， n 图片主名 f 处理方法名 l1 第一层 l2 第二层
        # self._p, self.n, self.f, self.l1, self.l2, self.skip, self.increase = None, n, "_", 0, 0, False, 1

    @property
    def p1(self):
        return self._p

    @p1.setter
    def p1(self, p):
        self._p = p
        self.l1 += 1
        if self.l2 > 0:
            self.f = ""
        self.l2 = 0
        self.write()

    @p1.deleter
    def p1(self):
        del self._p

    @property
    def p2(self):
        return self._p

    @p2.setter
    def p2(self, p):
        self._p = p
        self.l2 += 1
        self.write()

    @p2.deleter
    def p2(self):
        del self._p

    def write(self):
        if self.skip:
            pass
        else:
            # cv2.imwrite("output/tmp/%s.z.%s.%s.%s.png" % (self.n, str(self.l1).zfill(2), str(self.l2).zfill(2), self.f), self._p)
            cv2.imwrite("tmp/%s.z.%s.%s.%s.png" % (self.n, str(self.l1).zfill(2), str(self.l2).zfill(2), self.f),
                        self._p)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            print (anchors)
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        # if gpu_num>=2:
        #     self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
    #                     (image,prev_boxes,clock,car_number,Number,anchor_y)
    def detect_image(self, image,clock):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # self.boxes.extend(out_scores)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))#box is for target location
        # font = 'font/FiraMono-Medium.otf'
        font = ImageFont.truetype('/home/tsl/keras-yolo3/font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # font = ImageFont.truetype(,size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300
        # boxx=[]

        # image,line_up,line_down = tools.draw_line(image,distance=300)

        if (len(out_boxes)) != 0:
            out_classes, out_scores, out_boxes = tools.region_nms(out_classes,out_boxes,out_scores,image,clock)
            # tools.track_target(out_classes, out_scores, out_boxes,image,last2boxes=prev_boxes[-2:])
            get_num = tools.count(self,out_classes, out_scores, out_boxes, image)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # tools.region_nms(predicted_class,box,score)

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image, self.Number


    def detect_video(self,video_path):
        name = video_path.split('video/')[1].split('.mp4')[0]
        output_path = '/home/tsl/keras-yolo3/data/result/'+name+'.mp4'

        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(output_path, fourcc, 30.0, (1920,1080))
        accum_time, clock, curr_fps = 0,0,0
        fps = "FPS: ??"
        prev_time = timer()

        while True:
            ret, frame = vid.read()
            image = (Image.fromarray(frame)).rotate(-90)
            clock += 1
            image ,Number= YOLO.detect_image(self,image,clock)
            result = np.asarray(image)  #cv2.imshwo('result')
            _,w,h = result.shape[::-1]
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time % 5 == 0:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            #cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
            #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0,0,255), thickness=2)
            cv2.putText(result,text=str(Number),org=(int(w/2),int(h/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=10,color=(255,0,0),thickness=6)

            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)

            if ret:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if isOutput:
            out.write(result)

    def close_session(self):
        self.sess.close()


def main():
    video_path='./data/video/14.mp4'
    yolo = YOLO()
    yolo.detect_video(video_path)
    yolo.close_session()

if __name__ == '__main__':
    main()
