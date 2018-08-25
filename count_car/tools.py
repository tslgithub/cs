from PIL import Image, ImageFont, ImageDraw
import cv2
import os
import time
import time
import numpy as np
import operator

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
    Name = os.path.splitext(img)[0]
    img_name = Time+Name
    check_dir('./tmp')
    cv2.imwrite('./tmp/'+img_name,img)

def pil_save(img):
    Time = time.asctime(time.localtime(time.time())).split(':')[2].split(' ')[0]
    Name = os.path.splitext(img)[0]
    img_name = Time + Name
    img.save('./tmp/'+img_name+'.jpg', 'jpeg')

def region_nms(out_classes,out_boxes,out_scores,image):
    image = np.asarray(image)
    same_target_index = []
    delta = 0.7
    for i ,location_i in list(enumerate(list(out_boxes)))[:-1]:
        # xi, yi, xii, yii = location_i[0], location_i[1], location_i[2],location_i[3]
        # location_i.reverse()
        xi, yi, xii, yii = location_i
        central_i_x ,central_i_y= (xi + xii)/2 , (yi+yii)/2
        delta_x , delta_y = delta * (xii - xi), delta * (yii - yi)
        for j, location_j in list(enumerate(list(out_boxes)))[1:]:
            # location_j.reverse()
            xj, yj, xjj, yjj = location_j
            central_j_x,central_j_y = xj + xjj/2, (yj + yjj)/2
            if abs(central_i_x - central_j_x) < delta_x and abs(central_i_y - central_j_y) < delta_y:
                same_target_index.extend([i,j])

    if len(same_target_index) == 0:
        return out_classes,out_scores,out_boxes

    same_target = set(same_target_index)
    check_out_scores = []
    for check_index in same_target:
        check_out_scores.append(out_scores[check_index])
    max_scores_index = list(out_scores).index(max(check_out_scores))
    for k in same_target:
        if k != max_scores_index:
            # out_classes.tolist().pop(k)
            # out_boxes.tolist().pop(k)
            # out_scores.tolist().pop(k)
            # np.delete(out_classes,k,axis=0)
            # np.delete(out_scores,k,axis=0)
            # np.delete(out_boxes,k,axis=0)
            out_classes[k]=out_classes[max_scores_index]
            out_scores[k]=out_scores[max_scores_index]
            out_boxes[k]=out_boxes[max_scores_index]

    # out_boxes = out_boxes.astype(int)
    # lenght = len(out_boxes)
    # if len(out_boxes) > 1:
    #     i=0
    #     out_boxes_list  = out_boxes.tolist()
    #     for box in out_boxes_list:
    #         if out_boxes[i] in  out_boxes:
    #             out_boxes_list[1:].index(out_boxes_list[i])

    # return out_classes,out_scores,out_boxes
    return np.array(out_classes),np.array(out_scores),np.array(out_boxes)

def track_target(out_classes, out_scores, out_boxes,image,last2boxes):
    # return
    print('out_scores = ',out_scores.tolist())
    print('out_boxes = ',out_boxes.tolist())
    print('***************************************out_classes = ',out_classes.tolist())
    return

    if len(last2boxes)==1:
        return out_classes, out_scores, out_boxes
    # last2boxes
    delta = 0.3
    image = np.asarray(image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    w,h = gray.shape[::-1]
    image_line = cv2.line(image,(0,int(h/2)),(w,int(h/2)),(0,0,255),1)
    box_position = out_boxes.tolist()
    anchor ,wh, test_box_w, test_box_h= [], [], [], []
    for box in box_position:
        [x1, y1], [x2, y2] = box[:2][::-1], box[2:][::-1]
        cw1, ch1 = (x1 + x2) / 2, (y1 + y2) / 2
        anchor.append([cw1,ch1])
        wh.append([x2 - x1,y2 - y1])
        test_box_w.extend(cw1)
        test_box_h.extend(ch1)



    anc1,anc2 = anchor
    if anc2[0] - anc1[0] <  delta * wh[1][0] and  anc2[1] - anc2[1] < delta * wh[1][0] :
        pass

