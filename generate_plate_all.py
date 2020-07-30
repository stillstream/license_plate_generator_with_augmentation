import numpy as np
import cv2
import os
import glob
import math


from plate_number import random_select, generate_plate_number_white, generate_plate_number_yellow_xue
from plate_number import generate_plate_number_black_gangao, generate_plate_number_black_shi, generate_plate_number_black_ling
from plate_number import generate_plate_number_blue, generate_plate_number_yellow_gua
from plate_number import letters, digits, provinces

from all_plate_number_rule import LicensePlateNoGenerator
from plate_elements import LicensePlateElements
from image_augment import ImageAugmentation
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch

def gama_exc(x, r):
    x = np.float(x/255.0)
    x = np.power(x, r)*255.0
    x = x.astype(np.uint8)
    return x

def log_contrast(x, gain):
    x = np.float(x/255.0) + 1
    x = 255.0*gain*math.log(x, 2)
    x = np.uint8(x)
    return x

def light_change_right_trap_multi(images, random_state, parents, hooks):

    """
    任意尺寸直角梯形变换
    :param img: 输入图像
    :param flag: ’x‘ or 'y'
    :param gain: 变换率（降低）
    :param gain_: 变换率（升高）
    :return: 亮度调整后的图像
    """
    flag = 'x'
    gain = 0.25
    gain_ = 0.9
    new_images = list()
    for img in images:
        h, w = img.shape[0], img.shape[1]
        h1 = int(np.random.randint(0, h, 1))
        h2 = int(np.random.randint(0, h, 1))
        w1 = int(np.random.randint(0, w, 1))
        w2 = int(np.random.randint(0, w, 1))
        for x in range(0, w):
            for y in range(0, h):
                if 0 <= y <= ((h2 - h1) / w) * x + h1 and flag == 'y':
                    img[y, x, 0] = log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = log_contrast(img[y, x, 2], gain)
                elif 0 <= x <= (y * (w1 - w2) / h) + w2 and flag == 'x':
                    img[y, x, 0] = log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = log_contrast(img[y, x, 2], gain)
                else:
                    img[y, x, 0] = gama_exc(img[y, x, 0], gain_)
                    img[y, x, 1] = gama_exc(img[y, x, 1], gain_)
                    img[y, x, 2] = gama_exc(img[y, x, 2], gain_)
        new_images.append(img)
    return new_images

def get_location_data(length=7, split_id=1, height=140, width=440, green=False):
    """
    根据车牌规则来确定每个字符在车牌上的排布
    :param length: 车牌号码长度
    :param split_id: 分隔符
    :param height: 车牌高度
    :param width:  车牌宽度
    :param green: 是否为新能源车
    :return: 每个字符的左上和右下坐标
    """
    location_xy = np.zeros((length, 4), dtype=np.int32)
    # 加入二轮摩托车的规则
    if width == 220 and length == 7:
        location_xy[0, :] = [45, 10, 95, 60]
        location_xy[1, :] = [125, 10, 175, 60]
        width_font = 30
        step_font = 10
        for i in range(2,length):
            location_xy[i, 1] = 70
            location_xy[i, 3] = 130
            if i == 2:
                location_xy[i, 0] = 15
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font
    # 加入6位使馆摩托车规则
    elif width == 220 and length == 6:
        location_xy[0, :] = [70, 10, 150, 60]
        width_font = 30
        step_font = 10
        for i in range(1, length):
            location_xy[i, 1] = 70
            location_xy[i, 3] = 130
            if i == 1:
                location_xy[i, 0] = 15
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font
    # 加入香港本地车牌规则
    elif width == 440 and length == 6:
        location_xy[:, 1] = 20
        location_xy[:, 3] = 120
        step_split = 60
        step_font = 10
        width_font = 50
        for i in range(length):
            if i == 0:
                location_xy[i, 0] = 20
            elif i == split_id:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font
    # 加入澳门本地车牌规则
    elif width == 520 and length == 6:
        location_xy[:, 1] = 10
        location_xy[:, 3] = 110
        step_split = 39
        #step_split2 = 35
        step_font = 8
        width_font = 65
        for i in range(length):
            if i == 0:
                location_xy[i, 0] = 15
            elif i == split_id:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_split
            elif i == 4:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font

    elif width == 300 and length == 7:
        location_xy[0, :] = [70, 10, 125, 55]
        location_xy[1, :] = [175, 10, 230, 55]
        width_font = 45
        step_font = 12
        for i in range(2, length):
            location_xy[i, 1] = 65
            location_xy[i, 3] = 155
            if i == 2:
                location_xy[i, 0] = 14
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font
    elif width == 300 and length == 8:
        location_xy[0, :] = [71, 10, 116, 55]
        location_xy[1, :] = [170, 10, 200, 55]
        location_xy[2, :] = [205, 10, 235, 55]
        width_font = 45
        step_font = 12
        for i in range(3, length):
            location_xy[i, 1] = 65
            location_xy[i, 3] = 155
            if i == 3:
                location_xy[i, 0] = 14
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font

    elif (height == 140 and width == 440) or green:
        location_xy[:, 1] = 25
        location_xy[:, 3] = 115
        step_split = 34 if length == 7 else 49     # step_split是有间隔符的地方间隔宽度
        step_font = 12 if length == 7 else 9

        width_font = 45
        for i in range(length):
            if i == 0:
                location_xy[i, 0] = 15
            elif i == split_id:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            if length == 8 and i > 0:
                width_font = 43
            location_xy[i, 2] = location_xy[i, 0] + width_font
    elif height == 220 and width == 440:
        location_xy[0, :] = [110, 15, 190, 75]
        location_xy[1, :] = [250, 15, 330, 75]

        width_font = 65
        step_font = 15
        for i in range(2, length):
            location_xy[i, 1] = 90
            location_xy[i, 3] = 200
            if i == 2:
                location_xy[i, 0] = 27
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font

    return location_xy

def get_location_data_WJ(length=7, split_id=2, height=140):
    location_xy = np.zeros((length, 4), dtype=np.int32)
    if length == 7 and height == 140:
        location_xy[:, 1] = 25
        location_xy[:, 3] = 115
        step_font = 5
        step_split = 90     # 武警总牌的间隔60

        width_font = 45
        for i in range(length):
            if i == 0:
                location_xy[i, 0] = 15
            elif i == 1:
                location_xy[i, 0] = 15 + width_font
            elif i == split_id:
                location_xy[i, 0] = location_xy[i-1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i-1, 2] + step_font

            if i == 1:
                location_xy[i, 2] = location_xy[i, 0] + 30
            else:
                location_xy[i, 2] = location_xy[i, 0] + width_font
    elif length == 8 and height == 140:
        location_xy[:, 1] = 25
        location_xy[:, 3] = 115
        step_font = 5
        step_split = 40
        width_font = 45
        for i in range(length):
            if i == 0:
                location_xy[i, 0] = 15
            elif i == 1:
                location_xy[i, 0] = location_xy[i-1, 2]
            elif i == 2:
                location_xy[i, 0] = location_xy[i-1, 2] + 10
            elif i == split_id:
                location_xy[i, 0] = location_xy[i-1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i-1, 2] + step_font

            if i == 1:
                location_xy[i, 2] = location_xy[i, 0] + 30
            elif i == 2:
                location_xy[i, 2] = location_xy[i, 0] + 40
            else:
                location_xy[i, 2] = location_xy[i, 0] + width_font
    else:
        location_xy[0, :] = [100, 25, 155, 75]
        location_xy[1, :] = [155, 25, 180, 75]
        location_xy[2, :] = [260, 25, 340, 75]

        width_font = 65
        step_font = 15
        for i in range(3, length):
            location_xy[i, 1] = 90
            location_xy[i, 3] = 200
            if i == 3:
                location_xy[i, 0] = 27
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font
    return location_xy

def copy_to_image_multi(img, font_img, bbox, bg_color, is_red):
    x1, y1, x2, y2 = bbox
    font_img = cv2.resize(font_img, (x2 - x1, y2 - y1))
    img_crop = img[y1: y2, x1: x2, :]

    if is_red:
        img_crop[font_img < 200, :] = [0, 0, 255]
    elif 'blue' in bg_color or 'black' in bg_color or 'ling' in bg_color or \
            'shi' in bg_color or 'dishu_farm' in bg_color or 'macau' in bg_color or 'avail' in bg_color:
        img_crop[font_img < 200, :] = [255, 255, 255]
    else:
        img_crop[font_img < 200, :] = [0, 0, 0]
    return img

class MultiPlateGenerator:
    def __init__(self, adr_plate_model, adr_font, width, bg_color):
        self.adr_plate_model = adr_plate_model
        self.adr_font = adr_font
        #作为判断摩托车的条件
        self.width = width
        self.bg_color = bg_color
        if 'green' in bg_color:
            green = True
        else:
            green = False

        self.font_imgs = {}
        font_filenames = glob.glob(os.path.join(adr_font, '*.jpg'))
        for font_filename in font_filenames:
            font_img = cv2.imdecode(np.fromfile(font_filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            # print(font_filename)
            if '140' in font_filename:
                font_img = cv2.resize(font_img, (45, 90))
            elif '220' in font_filename:
                font_img = cv2.resize(font_img, (65, 110))    # 单双行的处理方法一致？？？
            elif font_filename.split('_')[-1].split('.')[0] in letters + digits:
                font_img = cv2.resize(font_img, (43, 90))
            elif 'motor' in font_filename:
                font_img = cv2.resize(font_img, (30, 60))
            self.font_imgs[os.path.basename(font_filename).split('.')[0]] = font_img

        self.location_xys = dict()          # 根据车牌长度、分隔点与高度来确定每一个字符对应的坐标矩阵
        if 'army' in bg_color:
            for i in [7, 8]:
                for j in [2, 3]:
                    for k in [140, 220]:
                        self.location_xys['{}_{}_{}'.format(i, j, k)] = \
                            get_location_data_WJ(length=i, split_id=j, height=k)
        else:
            for i in [6, 7, 8]:
                for j in [1, 2, 3, 4]:
                    for k in [140, 220, 165, 120]:
                        self.location_xys['{}_{}_{}'.format(i, j, k)] = \
                            get_location_data(length=i, split_id=j, height=k, width=width, green=green)

    def get_location_multi(self, plate_number, height=140):
        """
        确定车牌号的分隔符在那
        :param plate_number:车牌号
        :param height: 车牌高度
        :return: 车牌字符的坐标
        """
        length = len(plate_number)
        if '警' in plate_number:
            split_id = 1
        elif '使' in plate_number:
            if len(plate_number) == 7:
                split_id = 4
            else:
                split_id = 1
        else:
            split_id = 2
        return self.location_xys['{}_{}_{}'.format(length, split_id, height)]

    def get_location_multi_WJ(self, plate_number, height=140):
        length = len(plate_number)
        if length == 8 and height == 140:
            split_id = 3
        elif length == 8 and height == 220:
            split_id = 2
        else:               #考虑是否可以删除
            split_id = 2
        return  self.location_xys['{}_{}_{}'.format(length, split_id, height)]

    def get_location_multi_old(self, plate_number, height=140):
        length = len(plate_number)
        if '使' in plate_number:
            split_id = 3
        else:
            split_id = 4
        return self.location_xys['{}_{}_{}'.format(length, split_id, height)]

    def generate_plate_number(self):
        """
        该函数通过给出的车牌号来确定单双行、车牌底颜色
        :return: plate_number--车牌号
                 bg_color--车牌底颜色
                 is_double--单双行
        """
        rate = np.random.random(1)
        if rate > 0.4:     #蓝牌大于0.4
            plate_number = generate_plate_number_blue(length=random_select([7, 8]))
        else:
            #这部分的作用是？
            generate_plate_number_funcs = [generate_plate_number_white,
                                           generate_plate_number_yellow_xue,
                                           generate_plate_number_yellow_gua,
                                           generate_plate_number_black_gangao,
                                           generate_plate_number_black_shi,
                                           generate_plate_number_black_ling]
            plate_number = random_select(generate_plate_number_funcs)()

        bg_color = random_select(['blue'] + ['yellow'])       #只考虑蓝底与黄底？

        if len(plate_number) == 8:
            bg_color = random_select(['green_car'] * 10 + ['green_truck'])
        elif len(set(plate_number) & set(['使', '领', '港', '澳'])) > 0:
            bg_color = 'black'
        elif '警' in plate_number or plate_number[0] in letters:
            bg_color = 'white'
        elif len(set(plate_number) & set(['学', '挂'])) > 0:
            bg_color = 'yellow'
        is_double = random_select([False] + [True] * 3)

        if '使' in plate_number:
            bg_color = 'black_shi'

        if '挂' in plate_number:
            is_double = True
        elif len(set(plate_number) & set(['使', '领', '港', '澳', '学', '警'])) > 0 \
                or len(plate_number) == 8 or bg_color == 'blue':
            is_double = False

        # special
        if plate_number[0] in letters and not is_double:
            bg_color = 'white_army'

        return plate_number, bg_color, is_double

    def generate_plate_old(self, plate_number, bg_color, is_double):

        height = 220 if is_double else 140
        number_xy = self.get_location_multi_old(plate_number, height)
        img_plate_model = cv2.imread(os.path.join(self.adr_plate_model, '{}_{}.PNG'.format(bg_color, height)))
        img_plate_model = cv2.resize(img_plate_model, (440 if len(plate_number) == 7 else 480, height))

        for i in range(len(plate_number)):
            if len(plate_number) == 8:
                font_img = self.font_imgs['green_{}'.format(plate_number[i])]
            else:
                if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                    font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                else:
                    if i < 2:
                        font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                    else:
                        font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]
            is_red = False
            img_plate_model = copy_to_image_multi(img_plate_model, font_img,
                                                  number_xy[i, :], bg_color, is_red)

        # is_double = 'double' if is_double else 'single'
        img_plate = cv2.blur(img_plate_model, (3, 3))
        augment = ImageAugmentation()
        img_plate = augment.aug_combination(img_plate)
        return img_plate

    def generate_plate_special(self, plate_number, bg_color, is_double, enhance=False):
        """
        生成特定号码、颜色车牌
        :param plate_number: 车牌号码
        :param bg_color: 背景颜色
        :param is_double: 是否双层
        :param enhance: 图像增强
        :return: 车牌图
        """
        if 'motor' in bg_color:
            height = 140
            width = 220
        elif 'dishu' in bg_color:
            height = 165
            width = 300
        elif 'macau' in bg_color:
            height = 120
            width = 520
        else:
            height = 220 if is_double else 140
        print('车牌号是：{}'.format(plate_number), "\n",
              '车牌高度是：{}'.format(height), "\n",
              '车牌底颜色是：{}'.format(bg_color), "\n",
              '是否双行：{}'.format(is_double))
        if "army" in bg_color:
            number_xy = self.get_location_multi_WJ(plate_number, height)
        else:
            number_xy = self.get_location_multi(plate_number, height)
        print(number_xy)
        img_plate_model = cv2.imread(os.path.join(self.adr_plate_model, '{}_{}.PNG'.format(bg_color, height)))
        print(img_plate_model.shape)
        if "motor" in bg_color or 'dishu' in bg_color or 'macau' in bg_color:
            img_plate_model = cv2.resize(img_plate_model, (width, height))
        else:
            img_plate_model = cv2.resize(img_plate_model, (480 if 'green' in bg_color else 440, height))
        print(img_plate_model.shape)
        for i in range(len(plate_number)):
            if len(plate_number) == 8 and 'green' in bg_color:
                font_img = self.font_imgs['green_{}'.format(plate_number[i])]
            # 加入低速车车牌
            elif 'dishu' in bg_color:
                if 'farm' in bg_color:
                    if i == 0:
                        font_img = self.font_imgs['motor_up_{}'.format(plate_number[i])]
                    elif i == 1 or i == 2:
                        font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                        font_img = cv2.resize(font_img, (30, 45))
                    else:
                        font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                else:
                    if i == 0:
                        font_img = self.font_imgs['220_{}'.format(plate_number[i])]
                        font_img = cv2.resize(font_img, (55, 45))
                    elif i == 1:
                        font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                        font_img = cv2.resize(font_img, (55, 45))
                    else:
                        font_img = self.font_imgs['140_{}'.format(plate_number[i])]
            # 加入澳门车牌
            elif 'macau' in bg_color:
                font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                font_img = cv2.resize(font_img, (65, 100))
            # 加入武警单行地方车牌
            elif len(plate_number) == 8 and ('army' in bg_color and is_double == False):
                if i == 1:
                    font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                    font_img = cv2.resize(font_img, (30, 90))
                elif i == 2:
                    font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                    font_img = cv2.resize(font_img, (40, 90))
                else:
                    font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
            # 加入武警双行大车车牌
            elif 'army' in bg_color and is_double:
                if i < 3:
                    if i == 0:
                        font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                        font_img = cv2.resize(font_img, (55, 50))
                    if i == 1:
                        font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                        font_img = cv2.resize(font_img, (25, 50))
                    if i == 2:
                        font_img = self.font_imgs['220_{}'.format(plate_number[i])]
                        font_img = cv2.resize(font_img, (80, 50))
                else:
                    if plate_number[i] in digits:
                        font_img = self.font_imgs['220_{}'.format(plate_number[i])]
                    else:
                        font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]
            else:
                if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                    # 更改WJ中J的尺寸，武警小车车牌
                    if 'army' in bg_color and i == 1:
                        font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                        font_img = cv2.resize(font_img, (30, 90))
                    else:
                        font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                # 加入摩托车
                elif '{}_{}'.format(bg_color, plate_number[i]) in self.font_imgs:
                    if len(plate_number) == 7:
                        if i < 2:
                            font_img = self.font_imgs['motor_up_{}'.format(plate_number[i])]
                        else:
                            font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                    else:
                        if i < 1:
                            font_img = self.font_imgs['motor_up_{}'.format(plate_number[i])]
                            font_img = cv2.resize(font_img, (80, 50))
                        else:
                            font_img = self.font_imgs['140_{}'.format(plate_number[i])]

                else:
                    if i < 2:
                        font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                    else:
                        font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]
            if plate_number[i] in ['警', '使', '领'] or ('army' in bg_color and i == 0) or ('army' in bg_color and i == 1):
                is_red = True
            # 加入武警变红规则
            elif plate_number[i] in provinces and 'army' in bg_color:
                is_red = True
            # 加入军牌变红规则
            elif 'jun' in bg_color and (i == 0 or i == 1):
                is_red = True
            elif plate_number[i] in letters and (i == 6 or i == 7) and 'army' in bg_color:
                is_red = True
            else:
                is_red = False
            if enhance:
                k = np.random.randint(1, 6)
                kernel = np.ones((k, k), np.uint8)
                if np.random.random(1) > 0.5:
                    font_img = np.copy(cv2.erode(font_img, kernel, iterations=1))
                else:
                    font_img = np.copy(cv2.dilate(font_img, kernel, iterations=1))

            img_plate = copy_to_image_multi(img_plate_model, font_img,
                                                  number_xy[i, :], bg_color, is_red)
        img_plate = cv2.blur(img_plate, (3, 3))
        augment = ImageAugmentation()
        img_plate = augment.perspective_transform_iaa(img_plate)
        img_plate = augment.gaussian_noise_iaa(img_plate)
        img_plate = augment.add_smudge(img_plate)

        return img_plate

    def generate_plate_multicore(self, plate_numbers, bg_color, is_double, enhance=False):
        """
        生成特定号码、颜色车牌
        :param plate_numbers: 车牌号码列表
        :param bg_color: 背景颜色
        :param is_double: 是否双层
        :param enhance: 图像增强
        :return: 车牌图
        """
        if 'motor' in bg_color:
            height = 140
            width = 220
        elif 'dishu' in bg_color:
            height = 165
            width = 300
        elif 'macau' in bg_color:
            height = 120
            width = 520
        else:
            height = 220 if is_double else 140

        plate_images = list()
        for plate_number in plate_numbers:
            print('车牌号是：{}'.format(plate_number), "\n",
                  '车牌高度是：{}'.format(height), "\n",
                  '车牌底颜色是：{}'.format(bg_color), "\n",
                  '是否双行：{}'.format(is_double))
            if "army" in bg_color:
                number_xy = self.get_location_multi_WJ(plate_number, height)
            else:
                number_xy = self.get_location_multi(plate_number, height)
            print(number_xy)
            img_plate_model = cv2.imread(os.path.join(self.adr_plate_model, '{}_{}.PNG'.format(bg_color, height)))
            print(img_plate_model.shape)
            if "motor" in bg_color or 'dishu' in bg_color or 'macau' in bg_color:
                img_plate_model = cv2.resize(img_plate_model, (width, height))
            else:
                img_plate_model = cv2.resize(img_plate_model, (480 if 'green' in bg_color else 440, height))
            print(img_plate_model.shape)
            for i in range(len(plate_number)):
                if len(plate_number) == 8 and 'green' in bg_color:
                    font_img = self.font_imgs['green_{}'.format(plate_number[i])]
                # 加入低速车车牌
                elif 'dishu' in bg_color:
                    if 'farm' in bg_color:
                        if i == 0:
                            font_img = self.font_imgs['motor_up_{}'.format(plate_number[i])]
                        elif i == 1 or i == 2:
                            font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                            font_img = cv2.resize(font_img, (30, 45))
                        else:
                            font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                    else:
                        if i == 0:
                            font_img = self.font_imgs['220_{}'.format(plate_number[i])]
                            font_img = cv2.resize(font_img, (55, 45))
                        elif i == 1:
                            font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                            font_img = cv2.resize(font_img, (55, 45))
                        else:
                            font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                # 加入澳门车牌
                elif 'macau' in bg_color:
                    font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                    font_img = cv2.resize(font_img, (65, 100))
                # 加入武警单行地方车牌
                elif len(plate_number) == 8 and ('army' in bg_color and is_double == False):
                    if i == 1:
                        font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                        font_img = cv2.resize(font_img, (30, 90))
                    elif i == 2:
                        font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                        font_img = cv2.resize(font_img, (40, 90))
                    else:
                        font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                # 加入武警双行大车车牌
                elif 'army' in bg_color and is_double:
                    if i < 3:
                        if i == 0:
                            font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                            font_img = cv2.resize(font_img, (55, 50))
                        if i == 1:
                            font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                            font_img = cv2.resize(font_img, (25, 50))
                        if i == 2:
                            font_img = self.font_imgs['220_{}'.format(plate_number[i])]
                            font_img = cv2.resize(font_img, (80, 50))
                    else:
                        if plate_number[i] in digits:
                            font_img = self.font_imgs['220_{}'.format(plate_number[i])]
                        else:
                            font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]
                else:
                    if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                        # 更改WJ中J的尺寸，武警小车车牌
                        if 'army' in bg_color and i == 1:
                            font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                            font_img = cv2.resize(font_img, (30, 90))
                        else:
                            font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                    # 加入摩托车
                    elif '{}_{}'.format(bg_color, plate_number[i]) in self.font_imgs:
                        if len(plate_number) == 7:
                            if i < 2:
                                font_img = self.font_imgs['motor_up_{}'.format(plate_number[i])]
                            else:
                                font_img = self.font_imgs['140_{}'.format(plate_number[i])]
                        else:
                            if i < 1:
                                font_img = self.font_imgs['motor_up_{}'.format(plate_number[i])]
                                font_img = cv2.resize(font_img, (80, 50))
                            else:
                                font_img = self.font_imgs['140_{}'.format(plate_number[i])]

                    else:
                        if i < 2:
                            font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                        else:
                            font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]
                if plate_number[i] in ['警', '使', '领'] or ('army' in bg_color and i == 0) or ('army' in bg_color and i == 1):
                    is_red = True
                # 加入武警变红规则
                elif plate_number[i] in provinces and 'army' in bg_color:
                    is_red = True
                # 加入军牌变红规则
                elif 'jun' in bg_color and (i == 0 or i == 1):
                    is_red = True
                elif plate_number[i] in letters and (i == 6 or i == 7) and 'army' in bg_color:
                    is_red = True
                else:
                    is_red = False
                if enhance:
                    k = np.random.randint(1, 6)
                    kernel = np.ones((k, k), np.uint8)
                    if np.random.random(1) > 0.5:
                        font_img = np.copy(cv2.erode(font_img, kernel, iterations=1))
                    else:
                        font_img = np.copy(cv2.dilate(font_img, kernel, iterations=1))

                img_plate = copy_to_image_multi(img_plate_model, font_img,
                                                number_xy[i, :], bg_color, is_red)
                img_plate = cv2.blur(img_plate, (3, 3))
            plate_images.append(img_plate)
        # 多核数据增强
        import time
        time_start = time.time()
        augment = ImageAugmentation()
        batches = [UnnormalizedBatch(images=plate_images)]
        # 自行定义数据增强的方法
        aug = iaa.Sequential([iaa.imgcorruptlike.Snow(severity=3),
                             iaa.GaussianBlur(0.5),
                             iaa.CropAndPad(px=(-10, 10)),
                             #iaa.Lambda(func_images=light_change_right_trap_multi)
                              ])

        batches_aug = list(aug.augment_batches(batches, background=True))
        images = augment.light_change_right_trap(batches_aug[0], flag='x')
        time_end = time.time()
        print("Augmentation done in %.2fs" % (time_end - time_start,))
        #img_plate = augment.gaussian_noise_iaa(img_plate)
        #img_plate = augment.add_smudge(img_plate)

        return images


if __name__ == '__main__':
    # 车牌类型
    plate_type_enum = {0: 'single_blue', 1: 'single_yellow', 2: 'double_yellow', 3: 'police', 4: 'single_yellow_learner',
                       5: 'hk', 6: 'macau', 7: 'army7', 8: 'black_shi', 9: 'black_ling', 10: 'dishu_st', 11: 'avail',
                       12: 'army_double', 13: 'small_new_energy', 14: 'big_new_energy', 15: 'motor_light', 16: 'single_black',
                       17: 'jun_double', 18: 'jun', 19: 'army8', 20: 'black_gang', 21: 'black_ao', 22: 'motor_st', 23: 'motor_police',
                       24: 'motor_shi', 25: 'motor_ling', 26: 'motor_learner', 27: 'double_yellow_trailer', 28: 'double_yellow_coach',
                       29: 'dishu_farm', 30: 'old_shi', 31: 'old_ling'}
    print("30 kinds license plate as follows: \n", plate_type_enum)
    print("please choose license plate you want to make")
    type_choice = input()
    plate_type = plate_type_enum[int(type_choice)]
    batch_size = 32
    iteration = 1
    # 车牌背景颜色与单双行
    if 'army7' in plate_type:
        bg_color = 'white_army_7'
        is_double = False
    elif 'army8' in plate_type:
        bg_color = 'white_army_8'
        is_double = False
    elif 'army_double' in plate_type:
        bg_color = 'white_army'
        is_double = True
    elif 'blue' in plate_type:
        bg_color = 'blue'
        is_double = False
    elif 'single_yellow' in plate_type:
        bg_color = 'yellow'
        is_double = False
    elif 'double_yellow' in plate_type:
        bg_color = 'yellow'
        is_double = True
    elif 'small_new_energy' in plate_type:
        bg_color = 'green_car'
        is_double = False
    elif 'big_new_energy' in plate_type:
        bg_color = 'green_truck'
        is_double = False
    elif 'jun' in plate_type:
        if 'double' in plate_type:
            bg_color = 'white_jun'
            is_double = True
        else:
            bg_color = 'white_jun'
            is_double = False
    elif 'police' == plate_type:
        bg_color = 'white'
        is_double = False
    elif 'black_gang' in plate_type or 'black_ao' in plate_type or 'black_ling' in plate_type:
        bg_color = 'black'
        is_double = False
    elif 'black_shi' in plate_type:
        bg_color = 'black_shi'
        is_double = False
    elif 'motor' in plate_type:
        if 'st' in plate_type or 'learner' in plate_type:
            bg_color = 'motor'
            is_double = True
        elif 'light' in plate_type:
            bg_color = 'motor_light'
            is_double = True
        elif 'ling' in plate_type:
            bg_color = 'motor_ling'
            is_double = True
        elif 'shi' in plate_type:
            bg_color = 'motor_shi'
            is_double = True
        elif 'police' in plate_type:
            bg_color = 'motor_police'
            is_double = True
    elif 'dishu' in plate_type:
        if 'farm' in plate_type:
            bg_color = 'dishu_farm'
            is_double = True
        else:
            bg_color = 'dishu'
            is_double = True
    elif 'avail' in plate_type:
        bg_color = 'avail'
        is_double = False
    elif 'hk' in plate_type:
        bg_color = 'hk'
        is_double = False
    elif 'macau' in plate_type:
        bg_color = 'macau'
        is_double = False
    elif 'old' in plate_type:
        bg_color = plate_type
        is_double = False
    elif 'single_black' in plate_type:
        bg_color = 'black'
        is_double = False

    if 'motor' in bg_color:
        width = 220
        # green = False
    elif 'green' in bg_color:
        width = 480
       # green = True
    elif 'dishu' in bg_color:
        width = 300
    elif 'macau' in bg_color:
        width = 520
    else:
        width = 440
        #green = False
    print('{} is making,please wait'.format(plate_type))
    for i in range(iteration):
        plate_generator = LicensePlateNoGenerator(plate_type=plate_type)
        plate_numbers = plate_generator.generate_license_plate_numbers(batch_size)
        generator = MultiPlateGenerator('D:\\keda_project\\plate-generator\\plate_model',
                                        'D:\\keda_project\\plate-generator\\font_model',
                                        width=width, bg_color=bg_color)
        multi_core = False
        if multi_core == False:
            # 单张图片进行处理
            for plate_number in plate_numbers:
                if 'old' in plate_type:
                    img = generator.generate_plate_old(plate_number, bg_color, is_double)
                else:
                    img = generator.generate_plate_special(plate_number, bg_color, is_double)
                save_path = "D:/keda_project/plate-generator/"
                save_path = os.path.join(save_path, plate_type)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imencode(".jpg", img)[1].tofile(os.path.join(save_path, "{}.jpg".format(plate_number)))
        else:
            # 多核处理数据增强
            images = generator.generate_plate_multicore(plate_numbers, bg_color, is_double)
            for i, plate_number in enumerate(plate_numbers):
                save_path = "D:/keda_project/plate-generator/"
                save_path = os.path.join(save_path, plate_type)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imencode(".jpg", images.images_aug[i])[1].tofile(os.path.join(save_path, "{}.jpg".format(plate_number)))
    print('{} have been done,please look up'.format(plate_type))

