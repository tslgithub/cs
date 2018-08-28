#coding:utf-8

from PIL import Image, ImageFont, ImageDraw
import cv2
import os
import time
import time
import numpy as np
import operator
import PIL
# from PIL import Image
# from PIL import ImageDraw

def sh(img):
    Image.show(img)

def sh2(img):
    cv2.imshow('result',img)
    cv2.waitKey()

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir('./'+dir)

def save_img(img):
    Time =time.asctime(time.localtime(time.time())).split(':')[2].split(' ')[0]
    # Name = os.path.splitext(img)[0]
    img_name = Time
    check_dir('./tmp')
    cv2.imwrite('./tmp/'+img_name,img)

# def pil_save(img):
#     Time = time.asctime(time.localtime(time.time())).split(':')[2].split(' ')[0]
#     Name = os.path.splitext(img)[0]
#     img_name = Time + Name
#     img.save('./tmp/'+img_name+'.jpg', 'jpeg')
#
# def create_lines(x1,y1,x2,y2,image):
#     '''绘制干扰线'''
#
#         # 起始点
#     begin = (x1,y1)
#     #结束点
#     end = (x2,y2)
#     ImageDraw.draw.line([begin, end], fill=(255, 0, 0))
#     # fill = ()

# def create_points():
#     '''绘制干扰点'''
#     chance = min(100, max(0, int(point_chance))) # 大小限制在[0, 100]
#
#     for w in xrange(width):
#         for h in xrange(height):
#             tmp = random.randint(0, 100)
#             if tmp > 100 - chance:
#                 draw.point((w, h), fill=(0, 0, 0))

def region_nms(out_classes,out_boxes,out_scores,image,clock):
    w,h = image.size[::-1]
    # image = np.asarray(image)
    same_target_indexs = []
    delta = 0.5
    clock+=1
    delta_x, delta_y = delta*w,delta*h
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%region_nms')
    ####################################################################
    #                y1,x1,y2,x2 = out_boexes   x:heng   y:zong
    ####################################################################
    # print(int(out_boxes.tolist()[0][0]), 'firstfirstfirstfirstfirstfirstfirstfirstfirst')

    for i ,location_i in (list(enumerate(list(out_boxes)))[:-1]):

        # if int(out_boxes.tolist()[0][0]) != 883:
        #     continue

        # xi, yi, xii, yii = location_i[0], location_i[1], location_i[2],location_i[3]
        # location_i.reverse()
        yi, xi, yii, xii = location_i
        # central_i_x ,central_i_y= (xi + xii)/2 , (yi+yii)/2
        delta_x , delta_y = delta * (xii - xi), delta * (yii - yi)
        for j, location_j in list(enumerate(list(out_boxes)))[1+i:]:
            # location_j.reverse()
            yj, xj, yjj, xjj = location_j
            # central_j_x,central_j_y = xj + xjj/2, (yj + yjj)/2

            # if  int(out_boxes.tolist()[0][0]) != 88:
            #     continue

            if (abs(xi - xj) < delta_x and abs(yi - yj) < delta_y) \
                    or (abs(xii - xjj)< delta_x and abs(yii - yjj)<delta_y):
                same_target_indexs.append([i, j])
                rm_index = i if out_scores[i] < out_scores[j] else j

                out_boxes[rm_index] = out_boxes[rm_index]*0
                out_classes[rm_index] = out_classes[rm_index]*0
                out_scores[rm_index] = out_scores[rm_index]*0

                # new_out_boxes = np.delete(out_boxes, rm_index,axis=0)
                # new_out_scores = np.delete(out_scores, rm_index,axis=0)
                # new_out_classes = np.delete(out_classes, rm_index,axis=0)

                # (out_boxes.tolist()).pop(rm_index)
                # (out_scores.tolist()).pop(rm_index)
                # (out_classes.tolist()).pop(rm_index)


            # if abs(central_i_x - central_j_x) < delta_x and abs(central_i_y - central_j_y) < delta_y:
            #     same_target_indexs.append([i,j])


    new_out_boxes, new_out_classes, new_out_scores = [],[],[]#np.array([]), np.array([]), np.array([])
    for i in range(len(out_boxes)):
        if np.mean(out_boxes[i])>0.01:
            new_out_boxes.append(out_boxes.tolist()[i])
        if np.mean((out_scores[i]))>0.01:
            new_out_scores.extend([out_scores.tolist()[i]])
        if np.mean(out_classes[i])>0.01:
            new_out_classes.extend([out_classes.tolist()[i]])
    # print(new_out_boxes.tolist(), '************************************************************')
    # print( 'len(same_target_indexs)  =======',len(same_target_indexs))
    # if len(same_target_indexs) == 0:
    #     return out_classes,out_scores,out_boxes
    #
    #
    #
    # for same_target in same_target_indexs:
    #     same_out_score = []
    #
    #
    #
    #     for i in same_target:
    #         same_out_score.append(out_scores[i])
    #     same_target.remove(out_scores.tolist().index((max(same_out_score))))
    #     for k in same_target:
    #         k=max(k-1,0)
    #         out_boxes = np.delete(out_boxes, k, axis=0)
    #         out_scores = np.delete(out_scores, k, axis=0)
    #         out_classes = np.delete(out_classes, k, axis=0)
    #         print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&delete: ',out_boxes.tolist())
    #         print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&delete: : ',out_scores.tolist())
    #         print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&delete: : ',out_classes.tolist())


    # same_target = set(same_target_index)
    # check_out_scores = []
    # for check_index in same_target:
    #     check_out_scores.append(out_scores[check_index])
    # max_scores_index = list(out_scores).index(max(check_out_scores))
    #
    #
    # for k in same_target:
    #     if k != max_scores_index:
    #         # out_classes.tolist().pop(k)
    #         # out_boxes.tolist().pop(k)
    #         # out_scores.tolist().pop(k)
    #         #0.10458893100030764
    #         # if out_scores.tolist()[0]!=0.10458893100030764:
    #         #     continue
    #         # if int(out_boxes[0][0]) != 123:
    #         #     continue
    #
    #         # if k < len(out_classes):
    #             # break
    #         out_boxes = np.delete(out_boxes, k, axis=0)
    #         out_scores = np.delete(out_scores,k,axis=0)
    #         out_classes = np.delete(out_classes, k, axis=0)
    #         # out_classes[k]=out_classes[max_scores_index]
    #         # out_scores[k]=out_scores[max_scores_index]
    #         # out_boxes[k]=out_boxes[max_scores_index]

    # out_boxes = out_boxes.astype(int)
    # lenght = len(out_boxes)
    # if len(out_boxes) > 1:
    #     i=0
    #     out_boxes_list  = out_boxes.tolist()
    #     for box in out_boxes_list:
    #         if out_boxes[i] in  out_boxes:
    #             out_boxes_list[1:].index(out_boxes_list[i])
    # print(out_boxes,'************************************************************')
    print('class------------------ ',new_out_classes)
    print('out_scores------------- ',new_out_scores)
    print('out_boxes-------------- ',new_out_boxes)
    print('clock ------------------',clock)
    return np.array(new_out_classes),np.array(new_out_scores),np.array(new_out_boxes)
    # return np.array(out_classes),np.array(out_scores),np.array(out_boxes)

def track_target(out_classes, out_scores, out_boxes,image,last2boxes):
    return
    print('out_scores = ',out_scores.tolist())
    print('out_boxes = ',out_boxes.tolist())
    print('***************************************out_classes = ',out_classes.tolist())
    # return

    if len(last2boxes)==1:
        return out_classes, out_scores, out_boxes
    # last2boxes
    delta = 0.3
    # image = np.asarray(image)
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    w,h = image.size[::-1]
    image_line = cv2.line(np.asarray(image),(0,int(h/2)),(w,int(h/2)),(0,0,255),1)
    box_position = out_boxes.tolist()
    anchor ,wh, test_box_w, test_box_h= [], [], [], []
    for box in box_position:
        y1,x1,y2,x2 = box_position
        cw1, ch1 = (x1 + x2) / 2, (y1 + y2) / 2
        anchor.append([cw1,ch1])
        wh.append([x2 - x1,y2 - y1])
        test_box_w.extend(cw1)
        test_box_h.extend(ch1)

    anc1,anc2 = anchor
    if anc2[0] - anc1[0] <  delta * wh[1][0] and  anc2[1] - anc2[1] < delta * wh[1][0] :
        pass

def count(out_classes, out_scores, out_boxes,image,number):
    w, h = image.size# image has rotated  90 degree

    # cv2.rectangle()
    # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
    #cv2.rectangle(img, (x, y), (x + w, y + h), (B, G, R), Thickness)
    # cv2.putText()
    # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
    #cv2.putText(img, text, (x, y), Font, Size, (B, G, R), Thickness)

    image_cv = np.asarray(image)
    for y1,x1,y2,x2 in out_boxes.tolist():
        cv2.rectangle(image_cv,(int(x1),int(y1)),(max(int(x2)-30,30),int(y2)),(255,0,0),3)

    distance = 300
    image_line_up   = cv2.line(np.asarray(image), (0, int(h / 2) - distance), (w, int(h / 2) - distance), (0, 0, 255), 1)
    image_line_down = cv2.line(image_line_up, (0, int(h / 2)), (w, int(h / 2)), (0, 0, 255), 1)

    print('sdfasdf')
    # return np.asarray(image_cv).astype(float)
    return np.asarray(image_line_down)