#coding:utf-8

from PIL import Image, ImageFont, ImageDraw
import cv2
import os
import time
import time
import numpy as np
import operator
import PIL
from PIL import Image,ImageDraw
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

                # else:
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
    for bb in new_out_boxes:
        anchor = ( int((bb[1]+bb[3])/2),int((bb[0]+bb[2])/2))
        image = cv2.circle(np.asarray(image), anchor, 10, (0, 0, 255), -1)

    print('class------------------ ',new_out_classes)
    print('out_scores------------- ',new_out_scores)
    print('out_boxes-------------- ',new_out_boxes)
    print('clock ------------------',clock)
    return np.array(new_out_classes),np.array(new_out_scores),np.array(new_out_boxes),Image.fromarray(image)
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

def draw_line(image,distance):
    distance = 0
    w,h = image.size
    # h = 1.8*h
    image_line_up = cv2.line(np.asarray(image), (0, int(h / 2) - distance), (w, int(h / 2) - distance), (0, 0, 255), 3)
    image_line_down = cv2.line(image_line_up, (0, int(h / 2)), (w, int(h / 2)), (255, 0, 0), 3)
    image_pil = Image.fromarray(image_line_down)
    line_up = int(h / 2) - distance
    line_down = int(h / 2)
    return image_pil,line_up,line_down


###   https://github.com/alex-drake/OpenCV-Traffic-Counter
def count(self,out_classes, out_scores, out_boxes,image,distance):
    w, h = image.size# image has rotated  90 degree
    # anchor = []
    Boxes = []
    delta = 0.5
    pix = 200
    for box in out_boxes.astype(int).tolist():
        y1, x1, y2, x2 = box
        # if ((x1 < 500 and y1 <200 ) and (x2 > 1000 and y2>900)) :
        if (x2 - x1)*(y2-y1) > 0.5*w*h:
            continue
        # if y2 - y1 < h/2 and y2 - y1 > h/2 - 200:
        Boxes.append(box)

    if len(Boxes) == 0:
        return self.Number
    if len(self.Boxes) == 0:
        self.Number += len(Boxes)
        self.Boxes.append(Boxes)
        return self.Number


    target_number = len(Boxes)
    # for crt_box in Boxes:
    #     y11, x11,y12,x12 = crt_box
    #     for pre_box in self.Boxes[-1]:
    #         y21, x21, y22, x22 = pre_box
    #         if y21< (y11+y12)/2 <y22 and x21<(x11+x12)/2<x22:
    #             target_number-=1
    # h=1.6*h
    # target_number = len(Boxes)
    for crt_box in Boxes:
        y11, x11, y12, x12 = crt_box
        for pre_box in self.Boxes[-1]:
            y21, x21, y22, x22 = pre_box
    #         if  h/2 <(y11+y12)/2 < h/2+distance:
    #             self.Number+=1

            # (0, int(h / 2) - 100), (w, int(h / 2) - distance)
    # self.Number+=target_number
        if not (((y21 < y11 and y11<y22) and (x12 - x21 < delta*w) ) or ((y21 < y11 and y11<y22) and (x12 - x22 < delta*w) )):

            if (y22>y12 > y21 and (x11 > x21 or x11<x22)) or (y22>y12>y21 and (x12 > x21 or x12<x22)):
                target_number -= 1
    self.Number += 1


            # if (y12 - y21 < delta*pix and (x11 - x21 < delta*pix or  x22 - x11 < delta*pix)) or \
            #         (y12 - y21 < delta*pix and (x12 - x21 < delta*pix or  x22 - x12 < delta*pix)):
            #     target_number -= 1
            #     self.Number+=1
                # cv2.putText(np.asarray(image), text=str(self.Number), org=(int(w / 2), int(h / 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=10, color=(255, 0, 0), thickness=6)
    # if target_number>0:
        # self.Number += target_number


        # overlap_y1,overlap_x1,overlap_y2,overlap_x2 = y21,x11,y12,x22

        # img1 = (np.asarray(image))[slice(y11,y12),slice(x11,x12)]
        # img2 = (np.asarray(image))[slice(y21,y22),slice(x21,x22)]
        # print( 'feature ==== ', feature(img1,img2) )

        # ow ,oh = x22 - x11, y12 - y21

        # if oh <y12-y21 and ow< x22 -x21:
        # if oh < 0:
        #     ow ,oh = 0, 0
        #     # self.Number += 1
        # if (ow * oh) < min( abs((y12 - y11) * (x12 - x11)),abs(( y22 - y21)*(x22 - x21)) ) * 0.5:
        #     self.Number += 1
    # self.image = image

    self.Boxes.append(Boxes)
    return self.Number

def count2(self,out_classes, out_scores, out_boxes,image):
    pass


# 均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (192, 1088), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(192):
        for j in range(108):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 差值感知算法
def dHash(img):
    # 缩放8*8
    img = cv2.resize(img, (192, 108), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(108):
        for j in range(192):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def feature(img1,img2):
    # img1 = cv2.imread('walk_m.jpg')
    # img2 = cv2.imread('walks1.jpg')
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    print(hash1)
    print(hash2)
    n_hash = cmpHash(hash1, hash2)
    # print('均值哈希算法相似度：', n)

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    print(hash1)
    print(hash2)
    d_hash= cmpHash(hash1, hash2)
    # print('差值哈希算法相似度：', n)
    return  n_hash,d_hash



