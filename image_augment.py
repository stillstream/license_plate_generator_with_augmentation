# -*- coding: utf-8 -*-
"""
Created on 2019/4/17
File augment_image
@author:ZhengYuwei
@ref: https://github.com/huxiaoman7/mxnet-cnn-plate-recognition, @author Huxiaoman
功能：
进行图像增强：透视变换、HSV变化、添加背景、高斯噪声、添加污渍(运动模糊、腐蚀膨胀、下采样、光照变化)
"""
import cv2
import numpy as np
import math
import os
import random
import imgaug
import imgaug.augmenters as iaa


class ImageAugmentation(object):
    """ 一些图像增强操作操作: 透视变换、HSV变化、添加背景、高斯噪声、污渍 """

    horizontal_sight_directions = ('left', 'mid', 'right')
    vertical_sight_directions = ('up', 'mid', 'down')

    def __init__(self):            # , template_image
        """ 一些图像增强参数的默认初值初始化
        :param plate_type: 车牌类型，用于字符颜色修正
        :param template_image: 车牌底牌图片
        """
        #self.plate_type = plate_type
        # 确定字符颜色是否应该为黑色
        # if plate_type in ['single_blue', 'black_shi', 'black_ling', 'dishu_st', 'motor_light', 'single_black',
        #                   'black_gang', 'black_ao', 'motor_shi', 'motor_ling','dishu_farm', 'avail']:
        #     # 字符为白色
        #     self.is_black_char = False
        # elif plate_type in ['single_yellow', 'double_yellow', 'single_yellow_learner', 'small_new_energy', 'big_new_energy', 'police',
        #                     'hk', 'army7', 'army_double', 'jun_double', 'jun', 'army8', 'motor_st', 'motor_police', 'motor_learner',
        #                     'double_yellow_trailer', 'double_yellow_coach']:
        #     # 字符为黑字
        #     self.is_black_char = True
        # else:
        #     raise ValueError('暂时不支持该类型车牌')
        #self.template_image = template_image
        # 透视变换
        self.angle_horizontal = 6
        self.angle_vertical = 6
        self.angle_up_down = 6
        self.angle_left_right = 5
        self.factor = 10
        # 色调，饱和度，亮度
        self.hue_keep = 0.8
        self.saturation_keep = 0.3
        self.value_keep = 0.2
        # 自然环境照片的路径列表
        self.env_data_paths = ImageAugmentation.search_file("background")
        # 高斯噪声level
        self.level = 1 + ImageAugmentation.rand_reduce(4)
        # 污渍
        self.smu = cv2.imread("images/smu.jpg")

    def sight_transfer(self, images, horizontal_sight_direction, vertical_sight_direction):
        """ 对图片进行视角变换
        :param images: 图片列表
        :param horizontal_sight_direction: 水平视角变换方向
        :param vertical_sight_direction: 垂直视角变换方向
        :return:
        """
        flag = 0
        img_num = len(images)
        # 左右视角
        if horizontal_sight_direction == 'left':
            flag += 1
            images[0], matrix, size = self.left_right_transfer(images[0], is_left=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        elif horizontal_sight_direction == 'right':
            flag -= 1
            images[0], matrix, size = self.left_right_transfer(images[0], is_left=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass
        # 上下视角
        if vertical_sight_direction == 'down':
            flag += 1
            images[0], matrix, size = self.up_down_transfer(images[0], is_down=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        elif vertical_sight_direction == 'up':
            flag -= 1
            images[0], matrix, size = self.up_down_transfer(images[0], is_down=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass

        # 左下视角 或 右上视角
        if abs(flag) == 2:
            images[0], matrix, size = self.vertical_tilt_transfer(images[0], is_left_high=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)

            images[0], matrix, size = self.horizontal_tilt_transfer(images[0], is_right_tilt=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        # 左上视角 或 右下视角
        elif abs(flag) == 1:
            images[0], matrix, size = self.vertical_tilt_transfer(images[0], is_left_high=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)

            images[0], matrix, size = self.horizontal_tilt_transfer(images[0], is_right_tilt=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass

        return images

    def up_down_transfer(self, img, is_down=True, angle=None):
        """ 上下视角，默认下视角
        :param img: 正面视角原始图片
        :param is_down: 是否下视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            angle = self.rand_reduce(self.angle_up_down)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_down:
            pts2 = np.float32([[interval, 0], [0, size_src[1]],
                               [size_src[0] - interval, 0], [size_src[0], size_src[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_src[1]],
                               [size_src[0], 0], [size_src[0] - interval, size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def left_right_transfer(self, img, is_left=True, angle=None):
        """ 左右视角，默认左视角
        :param img: 正面视角原始图片
        :param is_left: 是否左视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            angle = self.angle_left_right  # self.rand_reduce(self.angle_left_right)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_left:
            pts2 = np.float32([[0, 0], [0, size_src[1]],
                               [size_src[0], interval], [size_src[0], size_src[1] - interval]])
        else:
            pts2 = np.float32([[0, interval], [0, size_src[1] - interval],
                               [size_src[0], 0], [size_src[0], size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def vertical_tilt_transfer(self, img, is_left_high=True):
        """ 添加按照指定角度进行垂直倾斜(上倾斜或下倾斜，最大倾斜角度self.angle_vertical一半）
        :param img: 输入图像的numpy
        :param is_left_high: 图片投影的倾斜角度，左边是否相对右边高
        """
        angle = self.rand_reduce(self.angle_vertical)

        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])

        # 计算图片进行上下倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[1]))
        size_target = (int(math.cos((float(angle) / 180) * math.pi) * shape[1]), shape[0] + interval)
        # 目标图像上四个顶点的坐标
        if is_left_high:
            pts2 = np.float32([[0, 0], [0, size_target[1] - interval],
                               [size_target[0], interval], [size_target[0], size_target[1]]])
        else:
            pts2 = np.float32([[0, interval], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0], size_target[1] - interval]])

        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    def horizontal_tilt_transfer(self, img, is_right_tilt=True):
        """ 添加按照指定角度进行水平倾斜(右倾斜或左倾斜，最大倾斜角度self.angle_horizontal一半）
        :param img: 输入图像的numpy
        :param is_right_tilt: 图片投影的倾斜方向（右倾，左倾）
        """
        angle = self.rand_reduce(self.angle_horizontal)

        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])

        # 计算图片进行左右倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        size_target = (shape[1] + interval, int(math.cos((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_right_tilt:
            pts2 = np.float32([[interval, 0], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0] - interval, size_target[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_target[1]],
                               [size_target[0] - interval, 0], [size_target[0], size_target[1]]])

        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    @staticmethod
    def rand_reduce(val):
        return int(np.random.random() * val)

    def rand_perspective_transfer(self, img, factor=None, size=None):
        """ 添加投影映射畸变
        :param img: 输入图像的numpy
        :param factor: 畸变的参数
        :param size: 图片的目标尺寸，默认维持不变
        """
        if factor is None:
            factor = self.factor
        if size is None:
            size = (img.shape[1], img.shape[0])
        shape = size
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        # 目标图像上四个顶点的坐标
        pts2 = np.float32([[self.rand_reduce(factor), self.rand_reduce(factor)],
                           [self.rand_reduce(factor), shape[0] - self.rand_reduce(factor)],
                           [shape[1] - self.rand_reduce(factor), self.rand_reduce(factor)],
                           [shape[1] - self.rand_reduce(factor), shape[0] - self.rand_reduce(factor)]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # 利用投影映射矩阵，进行透视变换
        dst = cv2.warpPerspective(img, matrix, size)
        return dst, matrix, size

    def rand_hsv(self, img):
        """ 添加饱和度光照的噪声
        :param img: BGR格式的图片
        :return 加了饱和度、光照噪声的BGR图片
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 色调，饱和度，亮度
        hsv[:, :, 0] = hsv[:, :, 0] * (self.hue_keep + np.random.random() * (1 - self.hue_keep))
        hsv[:, :, 1] = hsv[:, :, 1] * (self.saturation_keep + np.random.random() * (1 - self.saturation_keep))
        hsv[:, :, 2] = hsv[:, :, 2] * (self.value_keep + np.random.random() * (1 - self.value_keep))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def rand_environment(self, img, env_data_paths=None):
        """ 添加自然环境的噪声
        :param img: 待加噪图片
        :param env_data_paths: 自然环境图片路径列表
        """
        if env_data_paths is None:
            env_data_paths = self.env_data_paths
        # 随机选取环境照片
        index = self.rand_reduce(len(env_data_paths))
        env = cv2.imread(env_data_paths[index])
        env = cv2.resize(env, (img.shape[1], img.shape[0]))
        # 找到黑背景，反转为白
        bak = (img == 0)
        for i in range(bak.shape[2]):
            bak[:, :, 0] &= bak[:, :, i]
        for i in range(bak.shape[2]):
            bak[:, :, i] = bak[:, :, 0]
        bak = bak.astype(np.uint8) * 255
        # 环境照片用白掩码裁剪，然后与原图非黑部分合并
        inv = cv2.bitwise_or(bak, env)
        img = cv2.bitwise_or(inv, img)
        return img

    def add_gauss(self, img, level=None):
        """ 添加高斯模糊
        :param img: 待加噪图片
        :param level: 加噪水平
        """
        if level is None:
            level = self.level
        return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

    def add_single_channel_noise(self, single):
        """ 添加高斯噪声
        :param single: 单一通道的图像数据
        """
        diff = 255 - single.max()
        noise = np.random.normal(0, 1 + self.rand_reduce(6), single.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = diff * noise
        noise = noise.astype(np.uint8)
        dst = single + noise
        return dst

    def add_noise(self, img):
        """添加噪声"""
        img[:, :, 0] = self.add_single_channel_noise(img[:, :, 0])
        img[:, :, 1] = self.add_single_channel_noise(img[:, :, 1])
        img[:, :, 2] = self.add_single_channel_noise(img[:, :, 2])
        return img

    def add_smudge(self, img, smu=None):
        """添加污渍"""
        if smu is None:
            smu = self.smu
        # 截取某一部分
        rows = self.rand_reduce(smu.shape[0] - img.shape[0])
        cols = self.rand_reduce(smu.shape[1] - img.shape[1])
        add_smu = smu[rows:rows + img.shape[0], cols:cols + img.shape[1]]
        img = cv2.bitwise_not(img)
        img = cv2.bitwise_and(add_smu, img)
        img = cv2.bitwise_not(img)
        return img

    def motion_blur_cv(self, image, degree=12, angle=45):
        """
        运动模糊
        :param image:
        :param degree:
        :param angle:
        :return: ndarray
        """
        image = np.array(image)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

    def gama_exc(self, x, r):
        """
        gama变换
        :param x:像素值
        :param r: 变换率，大于1，亮度降低，小于1则反之
        :return: 变换后的像素值
        """
        x = np.float(x / 255.0)
        x = np.power(x, r) * 255.0
        x = x.astype(np.uint8)
        return x

    def log_contrast(self, x, gain=0.25):
        """
        log变换
        :param x:像素值
        :param gain: 变换率，越小亮度越低
        :return: 变换后的像素值
        """
        x = np.float(x / 255.0) + 1
        x = 255.0 * gain * math.log(x, 2)
        x = np.uint8(x)
        return x

    def line_exc(self, x, v=0.2, b=0.2):
        """
        线性变换
        :param x:像素值
        :param v: 变换率
        :param b: 偏置
        :return: 变换后的值
        """
        x = x * v + b
        return x

    def light_change_trap(self, img, gain=0.25, gain_=0.9):
        """
        任意尺寸梯形亮度调整
        :param img: 输入图像
        :param gain: 变换率（降低）
        :param gain_: 变换率（升高）
        :return: 亮度调整后的图像
        """
        h, w = img.shape[0], img.shape[1]
        x0 = int(np.random.randint(0, int(w / 3), 1))
        d = int(np.random.randint(0, int((w - x0) / 2), 1))
        d1 = int(np.random.randint(0, int(w - 2 * d - x0), 1))
        h1 = int(np.random.randint(0, h, 1))
        for x in range(0, w):
            for y in range(0, h):
                if x0 <= x <= d + x0:
                    if 0 <= y <= (x - x0) * (h / d):
                        img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                        img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                        img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                elif d + x0 <= x <= d + x0 + d1:
                    img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                elif d + x0 + d1 <= x <= 2 * d + d1 + x0:
                    if (-h / d) * (x - x0 - 2 * d - d1) >= y >= 0:
                        img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                        img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                        img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                else:
                    img[y, x, 0] = self.gama_exc(img[y, x, 0], gain_)
                    img[y, x, 1] = self.gama_exc(img[y, x, 1], gain_)
                    img[y, x, 2] = self.gama_exc(img[y, x, 2], gain_)

        return img

    def light_change_right_trap(self, img, flag, gain=0.25, gain_=0.9):
        """
        任意尺寸直角梯形变换
        :param img: 输入图像
        :param flag: ’x‘ or 'y'
        :param gain: 变换率（降低）
        :param gain_: 变换率（升高）
        :return: 亮度调整后的图像
        """
        h, w = img.shape[0], img.shape[1]
        h1 = int(np.random.randint(0, h, 1))
        h2 = int(np.random.randint(0, h, 1))
        w1 = int(np.random.randint(0, w, 1))
        w2 = int(np.random.randint(0, w, 1))
        for x in range(0, w):
            for y in range(0, h):
                if 0 <= y <= ((h2 - h1) / w) * x + h1 and flag == 'y':
                    img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                elif 0 <= x <= (y * (w1 - w2) / h) + w2 and flag == 'x':
                    img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                else:
                    img[y, x, 0] = self.gama_exc(img[y, x, 0], gain_)
                    img[y, x, 1] = self.gama_exc(img[y, x, 1], gain_)
                    img[y, x, 2] = self.gama_exc(img[y, x, 2], gain_)
        return img

    def light_change_rec(self, img, flag, gain, gain_):
        """
        任意矩形尺寸亮度调整
        :param img: 输入图像
        :param flag: 'left','right','center','side'表示矩形在图像的区域
        :param gain: 变换率（降低）
        :param gain_: 变换率（升高）
        :return: 亮度调整后的图像
        """
        h, w = img.shape[0], img.shape[1]
        w1 = int(np.random.randint(0, w, 1))
        w2 = int(np.random.randint(w1, w, 1))
        gain = 0.25
        gain_ = 0.9
        for x in range(0, w):
            for y in range(0, h):
                if 0 <= x <= w1 and flag == 'left':
                    img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                elif w1 <= x <= w and flag == 'right':
                    img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                elif w1 <= x <= w2 and flag == 'center':
                    img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                elif (0 <= x <= w1 or w2 <= x <= w) and flag == 'side':
                    img[y, x, 0] = self.log_contrast(img[y, x, 0], gain)
                    img[y, x, 1] = self.log_contrast(img[y, x, 1], gain)
                    img[y, x, 2] = self.log_contrast(img[y, x, 2], gain)
                else:
                    img[y, x, 0] = self.gama_exc(img[y, x, 0], gain_)
                    img[y, x, 1] = self.gama_exc(img[y, x, 1], gain_)
                    img[y, x, 2] = self.gama_exc(img[y, x, 2], gain_)

        return img

    @staticmethod
    def search_file(search_path, file_format='.jpg'):
        """在指定目录search_path下，递归目录搜索指定尾缀的文件
        :param search_path: 指定的搜索目录，如：./2018年收集的素材并已校正
        :param file_format: 文件尾缀，如‘.jpg’
        :return: 该目录下所有指定尾缀文件的路径组成的list
        """
        file_path_list = []
        # 获取：1.父目录绝对路径 2.所有文件夹名字（不含路径） 3.所有文件名字
        for root_path, dir_names, file_names in os.walk(search_path):
            # 收集符合条件的文件名
            for filename in file_names:
                if filename.endswith(file_format):
                    file_path_list.append(os.path.join(root_path, filename))
        return file_path_list

    def aug_affine(self, img, scale=0.9, angle=6, translate_x=0.1, translate_y=0.1):
        aug = iaa.Affine(scale=scale, rotate=angle, translate_percent={'x': translate_x, 'y': translate_y})
        img = aug(images=[img])
        return img[0]

    def aug_wise_affine(self, img, scale=0.03):
        aug = iaa.PiecewiseAffine(scale=scale)
        img = aug(images=[img])
        return img[0]

    def trim(self, img, percent=0.1, flag='x'):
        if flag == 'x':
            aug = iaa.TranslateX(percent=percent)
            img = aug(images=[img])
        else:
            aug = iaa.TranslateY(percent=percent)
            img = aug(images=[img])
        return img[0]

    def overexposed(self, img, value=2.5):
        aug = iaa.MultiplyElementwise(value)
        img = aug(images=[img])
        return img[0]

    def low_contrast(self, img, value=3, sigma=2):
        seq = iaa.Sequential([iaa.imgcorruptlike.Contrast(value),
                              iaa.GaussianBlur(sigma=sigma)])
        img = seq(images=[img])
        return img[0]

    def erode_dilate(self, img, size=(3, 3)):
        kernel = np.ones(size, np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img

    def weather(self, img, severity=3, flag='rain'):
        if flag == 'rain':
            aug = iaa.imgcorruptlike.Spatter(severity=severity)
        elif flag == 'snow':
            aug = iaa.imgcorruptlike.Snow(severity=severity)
        elif flag == 'fog':
            aug = iaa.imgcorruptlike.Fog(severity=severity)
        elif flag == 'frost':
            aug = iaa.imgcorruptlike.Frost(severity=severity)
        else:
            print('暂不支持该类天气')
        img = aug(images=[img])
        return img[0]

    def motion_blur_iaa(self, img, severity=3):
        aug = iaa.imgcorruptlike.MotionBlur(severity=severity)
        img = aug(images=[img])
        return img[0]

    def gaussian_noise_iaa(self, img, severity=3):
        aug = iaa.imgcorruptlike.GaussianNoise(severity=severity)
        img = aug(images=[img])
        return img[0]

    def impluse_noise(self, img, severity=3):
        aug = iaa.imgcorruptlike.ImpulseNoise(severity=severity)
        img = aug(images=[img])
        return img[0]

    def perspective_transform_iaa(self, img, scale=0.08, keep_size=False):
        aug = iaa.PerspectiveTransform(scale=scale, keep_size=keep_size)
        img = aug(images=[img])
        return img[0]

    def aug_combination(self, img, horizontal_sight_direction=None, vertical_sight_direction=None):
        if horizontal_sight_direction is None:
            horizontal_sight_direction = ImageAugmentation.horizontal_sight_directions[random.randint(0, 2)]
        if vertical_sight_direction is None:
            vertical_sight_direction = ImageAugmentation.vertical_sight_directions[random.randint(0, 2)]

        img = self.sight_transfer([img], horizontal_sight_direction, vertical_sight_direction)
        img, _, _ = self.rand_perspective_transfer(img[0])
        img = self.rand_hsv(img)
        img = self.add_gauss(img)
        img = self.add_noise(img)
        img = self.add_smudge(img)
        return img

    def augment(self, img, horizontal_sight_direction=None, vertical_sight_direction=None):
        """ 综合上面的加载操作，进行全流程加噪
        :param img: 待加噪图片，传进来的图片是白底黑字
        :param horizontal_sight_direction: 水平视角方向
        :param vertical_sight_direction: 垂直视角方向
        :return: 加噪后的图片，numpy数组
        """
        if horizontal_sight_direction is None:
            horizontal_sight_direction = ImageAugmentation.horizontal_sight_directions[random.randint(0, 2)]
        if vertical_sight_direction is None:
            vertical_sight_direction = ImageAugmentation.vertical_sight_directions[random.randint(0, 2)]


        # 转为黑底白字
        img = cv2.bitwise_not(img)
        if not self.is_black_char:
            #img = cv2.bitwise_or(img, self.template_image)
            img = cv2.bitwise_not(img)
            # 基于视角的变换
            img = self.sight_transfer([img], horizontal_sight_direction, vertical_sight_direction)
            img = img[0]
            # 加上随机透视变换，这个其实可以不用
            img, _, _ = self.rand_perspective_transfer(img)
            #img = self.rand_environment(img)
            img = self.rand_hsv(img)
        else:
            # 对文字和底牌进行一样的透视操作
            img = self.sight_transfer([img], horizontal_sight_direction, vertical_sight_direction)
            img, matrix, size = self.rand_perspective_transfer(img[0])
            #template_image = cv2.warpPerspective(template_image, matrix, size)
            # 底牌加背景
            #template_image = self.rand_environment(template_image)
            # 转为白底黑字
            img = cv2.bitwise_not(img)
            # 底牌加车牌文字
            #img = cv2.add(img, template_image)#bitwise_and(img, template_image)
            img = self.rand_hsv(img)

        #img = self.add_gauss(img)
        #img = self.motion_blur(img)
        #img = self.add_noise(img)
        img = self.light_change_rec(img, flag='left')
        img = self.add_smudge(img)


        # aug_m = iaa.imgcorruptlike.MotionBlur(severity=5)
        # aug_p = iaa.PiecewiseAffine(scale=0.01)
        # aug_c = iaa.GammaContrast(1.6)
        #aug_l = iaa.LogContrast(gain=0.6, per_channel=False)
        # aug_auto_c = iaa.pillike.Autocontrast(20)
        # aug_s = iaa.imgcorruptlike.Snow(severity=2)
        # aug_r = iaa.imgcorruptlike.Spatter(severity=3)
        # aug_i = iaa.imgcorruptlike.ImpulseNoise(severity=3)
        # aug_g = iaa.imgcorruptlike.GaussianNoise(severity=2)
        # aug_m_e = iaa.MultiplyElementwise(0.1)
        # aug_A = iaa.Affine(scale=0.7, rotate=(-6, 6),
        #                    translate_percent={"x": (0, 0.2), "y": (0, 0.2)})
        # #aug = iaa.imgcorruptlike.Spatter(severity=2)
        #img1 = aug_l(images=[img])
        return img
