### Torch inference in c++


### Torch infrence in python

```
#!/usr/bin/python2.6
# -*- coding: utf-8 -*-

from PIL import Image
import cv2
import numpy as np
import torch
import json

# 内置类型颜色映射
COLOR = {0: [0, 0, 0],
         1: [0x32, 0x0b, 0x86],
         2: [0xa0, 0x00, 0x37],
         3: [0x0a, 0x00, 0xb6],
         4: [0x00, 0x60, 0x0f],
         5: [0x00, 0x4d, 0x40],
         6: [0x00, 0x5c, 0xb2],
         7: [0x4b, 0x63, 0x6e],
         8: [0x32, 0x19, 0x11],
         9: [0xc7, 0x5b, 0x37],
         10: [0xac, 0x08, 0x00],
         11: [0xff, 0xff, 0x00],
         12: [0x00, 0x3d, 0x00],
         13: [0x00, 0x64, 0xb7],
         }


class SegObject:

    def __init__(self, cls_id, cls_name, point_a, point_b, point_c, point_d):
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.points = [point_a, point_b, point_c, point_d]

    def json(self):
        ps = []
        for p in self.points:
            ps.append(p)
        js = {"name": self.cls_name, "id": self.cls_id, "points": ps}
        return json.load(js)


# 由于结果可能出现噪点，对分割结果腐蚀一次
def erode_image(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.erode(image, kernel)
    return dst


class SegmentationHandler:

    def __init__(self, model_path, class_info, paint=False):
        self.color_map = {}
        self.paint = paint
        for i in range(0, class_info):
            self.color_map[i] = {"name": class_info[i], "color": COLOR[i]}
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

        self.model = ""
        self.model = self.model.cuda()
        self.model.eval()
        checkpoint = torch.load(model_path)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        self.model.load_state_dict(state_dict)

    def paint_color(self, mm):
        src = np.zeros((mm.shape[0], mm.shape[1], 3), dtype=np.uint8)
        for y in range(0, mm.shape[0]):
            for x in range(0, mm.shape[1]):
                label = mm[y, x]
                src[y, x] = np.array(self.color_map[label]["color"])
        return src

    def detect_image(self, image_path):
        org_img = Image.open(image_path).convert('RGB')
        inputs = pre_process(org_img)
        inputs = torch.unsqueeze(inputs, 0)
        ret = self.model(inputs)
        infer_result = ret[0].argmax(axis=0)
        return self.segmentation(infer_result)

    # 对结果处理
    def segmentation(self, org_result):
        if self.paint:
            # 将单通道检测结果恢复3通道上色，利于观察
            color_img = self.paint_color(org_result)
        else:
            color_img = None
        objects = []
        # 按照类型依次对检测目标计算最小外接框
        for cls in range(1, len(self.color_map)):
            clone = org_result.copy()
            # 最小外接框 变换目标为高值255 背景黑色0
            clone[clone == cls] = 255
            clone[clone != 255] = 0
            clone = erode_image(clone)
            contours, hierarchy = cv2.findContours(clone, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                se_obj = SegObject(cls, self.color_map[cls]["name"], (box[0][0][0], box[0][0][1]),
                                   (box[0][1][0], box[0][1][1]), (box[0][2][0], box[0][2][1]),
                                   (box[0][3][0], box[0][3][1]))
                objects.append(se_obj)
                if self.paint:
                    color_img = cv2.drawContours(color_img, [box], 0, (255, 255, 255), 1)
                    for p in range(0, len(c)):
                        color_img = cv2.circle(color_img, (c[p][0][0], c[p][0][1]), 2, (255, 255, 255), -1)
        return objects, color_img

```
