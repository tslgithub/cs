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

# import count_vehicle_number

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from keras.utils import multi_gpu_model
gpu_num=1

import math
import logging

tracked_blobs = []
tracked_conts = []
t_retval = []
frame_no = 0

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.frames_seen = 0
        self.counted = False
        self.vehicle_dir = 0

    @property
    def last_position(self):
        return self.positions[-1]
    @property
    def last_position2(self):
        return self.positions[-2]

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0
        self.frames_seen += 1

    def draw(self, output_image):
        for point in self.positions:
            cv2.circle(output_image, point, 2, (0, 0, 255), -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, (0, 0, 255), 1)
# ============================================================================

class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.log = logging.getLogger("vehicle_counter")

        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.vehicle_LHS = 0
        self.vehicle_RHS = 0
        self.max_unseen_frames = 10
        self.frame_w = shape[1]
        self.frame_h = shape[0]

    @staticmethod
    def get_vector(a, b):#a lastposition
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values decrease in clockwise direction.
        """
        dx =float(b[0] - a[0])#min(float(b[0]-a[0] +3),1.0) #
        dy = float(b[1] - a[1])# min(float(b[1]-a[1])+3.0,1.0)

        distance = math.sqrt(dx ** 2 + dy ** 2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx / dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx / dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx / dy))
            else:
                angle = 180.0

        return distance, angle, dx, dy

    @staticmethod
    def is_valid_vector(a, b):
        # vector is only valid if threshold distance is less than 12
        # and if vector deviation is less than 30 or greater than 330 degs
        distance, angle, _, _ = a
        threshold_distance = 60#test 50 and right line is correct      #12.0
        return (distance <= threshold_distance)

    def update_vehicle(self, vehicle, matches):
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            contour, centroid = match

            # store the vehicle data
            vector = self.get_vector(vehicle.last_position, centroid)

            # only measure angle deviation if we have enough points
            if vehicle.frames_seen > 2:
                prevVector = self.get_vector(vehicle.last_position2, vehicle.last_position)
                angleDev = abs(prevVector[1] - vector[1])
            else:
                angleDev = 0

            b = dict(
                id=vehicle.id,
                center_x=centroid[0],
                center_y=centroid[1],
                vector_x=vector[0],
                vector_y=vector[1],
                dx=vector[2],
                dy=vector[3],
                counted=vehicle.counted,
                frame_number=frame_no,
                angle_dev=angleDev
            )
            # print('frame_no ===== ', b['frame_number'])
            tracked_blobs.append(b)

            # check validity
            if self.is_valid_vector(vector, angleDev):
                vehicle.add_position(centroid)
                vehicle.frames_seen += 1
                # check vehicle direction
                if vector[3] > 0:
                    # positive value means vehicle is moving DOWN
                    vehicle.vehicle_dir = 1
                elif vector[3] < 0:
                    # negative value means vehicle is moving UP
                    vehicle.vehicle_dir = -1
                self.log.debug("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)"
                               , centroid[0], centroid[1], vehicle.id, vector[0], vector[1])
                return i

        # No matches fit...
        vehicle.frames_since_seen += 1
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d"
                       , vehicle.id, vehicle.frames_since_seen)

        return None

    def update_count(self, matches, output_image=None):
        self.log.debug("Updating count using %d matches...", len(matches))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            self.log.debug("Created new vehicle #%d from match (%d, %d)."
                           , new_vehicle.id, centroid[0], centroid[1])

        for vehicle in self.vehicles:
            if not vehicle.counted and (((vehicle.last_position[0] < self.divider) and (vehicle.vehicle_dir == 1)) or
                                        ((vehicle.last_position[0] > self.divider) and (vehicle.vehicle_dir == -1)))\
                    and (vehicle.frames_seen > 6):
                    # and (vehicle.frames_seen > 6):

                vehicle.counted = True
                # frame_w = self.frame_w
                # frame_h = self.frame_h
                # update appropriate counter
                if ((vehicle.last_position[0] > self.divider - 300) and (vehicle.vehicle_dir == 1) and (
                        vehicle.last_position[1] <= (int(self.frame_w / 2) - 10))):
                    self.vehicle_LHS += 1
                    self.vehicle_count += 1
                    # pass
                elif (( vehicle.last_position[0] > self.divider) and (vehicle.vehicle_dir == -1) and (
                        vehicle.last_position[1] >= (int(self.frame_w / 2) + 10))):
                    self.vehicle_RHS += 1
                    self.vehicle_count += 1

                self.log.debug("Counted vehicle #%d (total count=%d)."
                               , vehicle.id, self.vehicle_count)

        # Count any uncounted vehicles that are past the divider
        # for vehicle in self.vehicles:
        #     if not vehicle.counted and (((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1)) or
        #                                 ((vehicle.last_position[1] < self.divider) and (
        #                                         vehicle.vehicle_dir == -1))) and (vehicle.frames_seen > 6):
        #
        #         vehicle.counted = True
        #         # frame_w = self.frame_w
        #         # frame_h = self.frame_h
        #         # update appropriate counter
        #         if ((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1) and (
        #                 vehicle.last_position[0] >= (int(self.frame_w / 2) - 10))):
        #             self.vehicle_RHS += 1
        #             self.vehicle_count += 1
        #         elif ((vehicle.last_position[1] < self.divider) and (vehicle.vehicle_dir == -1) and (
        #                 vehicle.last_position[0] <= (int(self.frame_w / 2) + 10))):
        #             self.vehicle_LHS += 1
        #             self.vehicle_count += 1
        #
        #         self.log.debug("Counted vehicle #%d (total count=%d)."
        #                        , vehicle.id, self.vehicle_count)

        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)

            # LHS
            cv2.putText(output_image, ("LH Lane: %02d" % self.vehicle_LHS), (12, 200)
                        , cv2.FONT_HERSHEY_PLAIN, 6, (127, 255, 255), 10)
            # RHS
            cv2.putText(output_image, ("RH Lane: %02d" % self.vehicle_RHS), (int(self.frame_w /2), 200)
                        , cv2.FONT_HERSHEY_PLAIN, 6, (127, 255, 255), 10)

        # Remove vehicles that have not been seen long enough
        removed = [v.id for v in self.vehicles
                   if v.frames_since_seen >= self.max_unseen_frames]
        self.vehicles[:] = [v for v in self.vehicles
                            if not v.frames_since_seen >= self.max_unseen_frames]
        for id in removed:
            self.log.debug("Removed vehicle #%d.", id)

        self.log.debug("Count updated, tracking %d vehicles.", len(self.vehicles))

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
        self.image=np.asarray([])
        self.frame_w = 1920
        self.frame_h = 1080

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

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))#box is for target location
        # font = 'font/FiraMono-Medium.otf'
        font = ImageFont.truetype('/home/tsl/keras-yolo3/font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # font = ImageFont.truetype(,size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300
        distance = 100
        # image,line_up,line_down = tools.draw_line(image,distance=distance)

        if (len(out_boxes)) != 0:
            out_classes, out_scores, out_boxes,_ = tools.region_nms(out_classes,out_boxes,out_scores,image,clock)

            # tools.track_target(out_classes, out_scores, out_boxes,image,last2boxes=prev_boxes[-2:])
            # get_num = tools.count(self,out_classes, out_scores, out_boxes, image,distance=distance)

            # count_vehicle_number.test_main(out_boxes,image,frame_w = 1920,frame_h=1080)

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
        print('time:    ',end - start)
        return image, self.Number,out_boxes

    def detect_video(self,video_path):
        # name = video_path.split('video/')[1].split('.mp4')[0]
        # output_path = '/home/tsl/keras-yolo3/data/result/'+name+'.mp4'
        output_path = '/home/tsl/keras-yolo3/data/result/'+'tmp.mp4'

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
        total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        total_cars = 0

        car_counter = None
        blobs = []
        ret, frame = vid.read()
        frame_no = 0
        ret, frame = vid.read()
        fps = int(video_fps + 1) * 2

        while ret:
            ret, frame = vid.read()

            frame_no = frame_no + 1
            # if frame_no % fps != 0:
            #     continue

            if ret and frame_no < total_frames:
                image = Image.fromarray(frame)
                #image = (Image.fromarray(frame)).rotate(-180)#.rotate(90).rotate(90)
                clock += 1
                image ,Number, boxes= YOLO.detect_image(self,image,clock)

                image = np.asarray(image)
                boxes=boxes.astype(int)
                for box in boxes:
                    x,y,w,h = box[1],box[0],box[3]-box[1],box[2]-box[0]
                    center = (int((box[1]+box[3])/2),int((box[0]+box[2])/2))
                    blobs.append(((x,y,w,h),center))

                for (i,match) in enumerate(blobs):
                    contour,centroid = match
                    c = dict(
                        frame_no = frame_no,
                        center_x = x,
                        center_y = y,
                        width = w,
                        height = h
                    )

                    tracked_conts.append(c)
                        # cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), LINE_THICKNESS)
                    cv2.circle(frame, centroid, 2, (0, 0, 255), -1)
                thresh_line = int(4 * self.frame_h / 5)
                if car_counter is None:
                    print("Creating vehicle counter...")
                    car_counter = VehicleCounter(image.shape[:2], thresh_line)

                car_counter.update_count(blobs, image)
                current_count = car_counter.vehicle_RHS + car_counter.vehicle_LHS
                # current_count =  car_counter.vehicle_LHS

                if current_count > total_cars:
                    cv2.line(image, (0, thresh_line), (self.frame_w, thresh_line),
                             (0, 255, 0), 2)
                else:
                    cv2.line(image, (0, thresh_line), (self.frame_w, thresh_line),
                             (0, 0, 255), 2)
                total_cars = current_count
                print('****'*20,total_cars)

                self.Number = total_cars
                result = image#np.asarray(image)  #cv2.imshwo('result')
                _,w,h = result.shape[::-1]
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1


                # if accum_time % 1 == 0:
                #     accum_time = accum_time - 1
                #     fps = "FPS: " + str(curr_fps)
                #     curr_fps = 0
                #cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
                #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
                # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.50, color=(0,0,255), thickness=2)
                cv2.putText(result,text=str(Number),org=(int(w/4),int(h/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
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
    # frame_no = 0
    # video_path='./data/video/1.mp4'
    video_path='/home/tsl/dataset/20180917/V80917-131752.mp4'
    # video_path='./9999.mp4'
    yolo = YOLO()
    yolo.detect_video(video_path)
    yolo.close_session()

if __name__ == '__main__':
    main()
