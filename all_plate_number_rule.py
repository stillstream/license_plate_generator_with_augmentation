# -*- coding: utf-8 -*-
"""
Created on 2019/4/17
File generate_license_plate_number
@author:ZhengYuwei
功能：
定制不同类型车牌的车牌号规则，生成指定数量的车牌号
"""
import numpy as np
from plate_elements import LicensePlateElements
from  plate_number import digits, letters, provinces


class LicensePlateNoGenerator(object):
    """ 随机生成车牌号和类型 """
    # 数字和英文字母列表
    numerals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, plate_type):
        """ 初始化随机生成的比例，用于后续的随机生成方法中
        :param plate_type: 需要生成的车牌类型
        """
        if plate_type not in LicensePlateElements.plate_type_enum.keys():
            print('车牌类型指定错误，请参考license_plate_elements,py里的plate_type_enum变量！')
            self.plate_type = None
            return
        # 车牌类型
        self.plate_type = plate_type
        # 车牌号元素对象
        self.elements = LicensePlateElements()

    def generate_license_plate_numbers(self, num):
        """ 生成指定数量的车牌号
        :param num: 数量
        :return: 车牌号及对应label
        """
        license_plate_numbers = None
        # 蓝色底牌、黄色底牌 车牌号为标准车牌规则
        if self.plate_type in ['single_blue', 'single_yellow', 'dishu_st', 'avail', 'single_black']:
            license_plate_numbers = self.generate_standard_license_no(num, self.plate_type)
        # 小型新能源车牌号规则
        elif self.plate_type == 'small_new_energy' or self.plate_type == 'dishu_farm':
            license_plate_numbers = self.generate_small_new_energy_license_no(num, self.plate_type)
        # 大型新能源车牌号规则
        elif self.plate_type == 'big_new_energy':
            license_plate_numbers = self.generate_big_new_energy_license_no(num)
        # 警车车牌号规则
        elif self.plate_type == 'police' or self.plate_type == 'single_yellow_learner':
            license_plate_numbers = self.generate_police_license_no(num, self.plate_type)
        # 大型汽车或者挂车或者教练车规则
        elif self.plate_type in ['double_yellow', 'double_yellow_trailer', 'double_yellow_coach']:
            license_plate_numbers = self.generate_double_yellow_license_no(num, self.plate_type)
        # 军区车车牌号规则
        elif self.plate_type == 'jun' or self.plate_type == 'jun_double':
            license_plate_numbers = self.generate_jun_license_no(num)
        # 武警车车牌号规则
        elif self.plate_type == 'army7':
            license_plate_numbers = self.generate_WJ_license_no(num, header=True)
        elif self.plate_type == 'army8' or self.plate_type == 'army_double':    # 双排与单排规则相同
            license_plate_numbers = self.generate_WJ_license_no(num, header=False)
        # 香港与澳门本地车牌号规则
        elif self.plate_type == 'hk' or self.plate_type == 'macau':
            license_plate_numbers = self.generate_gangao_license_no(num, self.plate_type)
        # 香港与澳门入大陆车牌号规则
        elif self.plate_type == 'black_gang' or self.plate_type == 'black_ao':
            license_plate_numbers = self.generate_black_gangao_license_no(num, self.plate_type)
        # 使馆与领馆车牌号规则
        elif self.plate_type == 'black_shi' or self.plate_type == 'black_ling':
            license_plate_numbers = self.generate_embra_license_no(num, self.plate_type)
        elif 'motor' in self.plate_type:
            if 'st' in self.plate_type or 'light' in self.plate_type:
                license_plate_numbers = self.generate_standard_license_no(num, self.plate_type)
            elif 'ling' in self.plate_type or 'shi' in self.plate_type:
                license_plate_numbers = self.generate_embra_license_no(num, self.plate_type)
            elif 'police' in self.plate_type:
                license_plate_numbers = self.generate_police_license_no(num, self.plate_type)
            elif 'learner' in self.plate_type:
                license_plate_numbers = self.generate_double_yellow_license_no(num, self.plate_type)
        elif 'old' in self.plate_type:
            license_plate_numbers = self.generate_old_license_no(num, self.plate_type)
        else:
            raise ValueError('暂时不支持该类型车牌！')

        return license_plate_numbers

    def generate_standard_license_no(self, num, plate_type):
        """ 生成蓝色、黄色等标准规则车牌号
        :param num: 生成车牌的数量
        :return: 生成的车牌号列表
        """
        # 针对车牌的每一位，随机采样
        license_chars = list()
        for char_index in range(7):
            # 对应车牌位上限制的字符范围
            char_range = self.elements.get_chars_sorted_by_label(char_index)
            if char_index == 0:
                if 'avail' in plate_type:
                    char_range = ['民']
                else:
                    char_range = char_range[:31]  # 第一位排除掉‘军’和‘使’
            elif char_index == 1:                # 第二位的范围还和省份相关，这里没考虑
                if 'avail' in plate_type:
                    char_range = ['航']
                else:
                    char_range = char_range[:24]  # 第二位排除数字
            elif char_index == 6:
                char_range = char_range[:34]  # 第六位排除‘学’、‘警’等特殊字符

            license_chars.append(np.random.choice(a=char_range, size=num, replace=True))

        # 取每一位，组成7位车牌
        license_plate_numbers = [list(_) for _ in zip(*license_chars)]
        # 在后五位编码可以出现字母，但不能超过两个
        for i, lic_no in enumerate(license_plate_numbers):
            # 找出后5位中英文字母的位置
            alphabet_loc = list()
            for loc in range(2, 7):
                if lic_no[loc] in LicensePlateNoGenerator.alphabet:
                    alphabet_loc.append(loc)
            # 字母数多于两个的，随机保留2个（这样会导致车牌中2位字母数的车牌比较多）
            if len(alphabet_loc) > 2:
                allow = np.random.choice(a=alphabet_loc, size=2, replace=False)
                alphabet_loc.remove(allow[0])
                alphabet_loc.remove(allow[1])

                # 多出来的字母，替换为数字
                new_nos = np.random.choice(a=LicensePlateNoGenerator.numerals, size=len(alphabet_loc), replace=True)
                for j, loc in enumerate(alphabet_loc):
                    lic_no[loc] = new_nos[j]

        license_plate_numbers = [''.join(_) for _ in license_plate_numbers]
        return license_plate_numbers

    def generate_small_new_energy_license_no(self, num, plate_type):
        """ 生成小型新能源车牌号
        :param num: 生成车牌的数量
        :return: 生成的车牌号列表
        """
        # 针对车牌的每一位，随机采样
        license_chars = list()
        for char_index in range(8):
            char_range = self.elements.get_chars_sorted_by_label(char_index)
            # 对应车牌位上限制的字符范围
            if char_index == 0:
                char_range = char_range[:31]  # 排除掉‘军’和‘使’
            elif char_index == 1:      # 第二位的范围还和省份相关，这里没考虑
                if 'dishu' in plate_type:
                    char_range = ['0']
                else:
                    char_range = char_range[:24]  # 第二位排除数字
            elif char_index == 2:
                if 'dishu' in plate_type:
                    char_range = ['4']
                else:
                    char_range = ['D', 'F']  # 小型新能源第3位为D或F
            elif char_index == 3:
                # 该位规则符合取值范围
                pass
            else:
                char_range = LicensePlateNoGenerator.numerals  # 小型新能源后4位必须用数值

            license_chars.append(np.random.choice(a=char_range, size=num, replace=True))

        # 取每一位，组成8位车牌
        license_plate_numbers = [''.join(_) for _ in zip(*license_chars)]
        return license_plate_numbers

    def generate_police_license_no(self, num, plate_type):
        """ 生成白底警用汽车车牌；
        :param num: 生成车牌的数量
        :return: 生成的车牌号列表"""
        # 针对车牌的每一位，随机采样
        license_chars = list()
        for char_index in range(7):
            char_range = self.elements.get_chars_sorted_by_label(char_index)
            # 对应车牌位上限制的字符范围
            if char_index == 0:
                char_range = char_range[:31]  # 排除掉‘军’和‘使’
            elif char_index == 1:
                # 第二位的范围还和省份相关，这里没考虑
                char_range = char_range[:24]  # 第二位排除数字
            elif char_index == 6:
                if 'police' in plate_type:
                    char_range = ['警']  # 警用车最后一位为警
                else:
                    char_range = ['学']
            license_chars.append(np.random.choice(a=char_range, size=num, replace=True))

        # 取每一位，组成7位车牌
        license_plate_numbers = [list(_) for _ in zip(*license_chars)]
        # 在后五位编码可以出现字母，但不能超过两个
        for i, lic_no in enumerate(license_plate_numbers):
            # 找出后5位中英文字母的位置
            alphabet_loc = list()
            for loc in range(2, 7):
                if lic_no[loc] in LicensePlateNoGenerator.alphabet:
                    alphabet_loc.append(loc)
            # 字母数多于两个的，随机保留2个（这样会导致车牌中2位字母数的车牌比较多）
            if len(alphabet_loc) > 2:
                allow = np.random.choice(a=alphabet_loc, size=2, replace=False)
                alphabet_loc.remove(allow[0])
                alphabet_loc.remove(allow[1])

                # 多出来的字母，替换为数字
                new_nos = np.random.choice(a=LicensePlateNoGenerator.numerals, size=len(alphabet_loc), replace=True)
                for j, loc in enumerate(alphabet_loc):
                    lic_no[loc] = new_nos[j]

        license_plate_numbers = [''.join(_) for _ in license_plate_numbers]
        return license_plate_numbers

    def generate_big_new_energy_license_no(self, num):
        """ 生成大型新能源汽车的车牌号
        :param num: 数量
        :return: 车牌号及对应label
        """
        # 针对车牌的每一位，随机采样
        license_chars = list()
        for char_index in range(8):
            char_range = self.elements.get_chars_sorted_by_label(char_index)
            # 对应车牌位上限制的字符范围
            if char_index == 0:
                char_range = char_range[:31]  # 排除掉‘军’和‘使’
            elif char_index == 1:
                # 第二位的范围还和省份相关，这里没考虑
                char_range = char_range[:24]  # 第二位排除数字
            elif char_index == 2:
                pass
            elif char_index == 3 or char_index == 4 or char_index == 5 or char_index == 6:
                char_range = LicensePlateNoGenerator.numerals  # 大型新能源后4位必须用数值
            elif char_index == 7:
                char_range = ['D', 'F']  # 大型新能源第8位为D或F

            license_chars.append(np.random.choice(a=char_range, size=num, replace=True))

        # 取每一位，组成8位车牌
        license_plate_numbers = [''.join(_) for _ in zip(*license_chars)]
        return license_plate_numbers

    def generate_double_yellow_license_no(self, num, plate_type):
        """ 生成黄底双行规则车牌号
        :param num: 生成车牌的数量
        :return: 生成的车牌号列表
        """
        # 针对车牌的每一位，随机采样
        license_chars = list()
        if plate_type in ['double_yellow_trailer', 'double_yellow_coach', 'motor_learner']:

            for char_index in range(7):
                # 对应车牌位上限制的字符范围
                char_range = self.elements.get_chars_sorted_by_label(char_index)
                if char_index == 0:
                    char_range = char_range[:31]  # 第一位排除掉‘军’和‘使’
                elif char_index == 1:
                    # 第二位的范围还和省份相关，这里没考虑
                    char_range = char_range[:24]  # 第二位排除数字
                elif char_index == 6:
                    if plate_type == 'double_yellow_trailer':
                        char_range = ['挂']
                    else:
                        char_range = ['学']

                license_chars.append(np.random.choice(a=char_range, size=num, replace=True))

                # 取每一位，组成7位车牌
            license_plate_numbers = [list(_) for _ in zip(*license_chars)]
            # 在后五位编码可以出现字母，但不能超过两个
            for i, lic_no in enumerate(license_plate_numbers):
                # 找出后5位中英文字母的位置
                alphabet_loc = list()
                for loc in range(2, 6):
                    if lic_no[loc] in LicensePlateNoGenerator.alphabet:
                        alphabet_loc.append(loc)
                # 字母数多于两个的，随机保留2个（这样会导致车牌中2位字母数的车牌比较多）
                if len(alphabet_loc) > 2:
                    allow = np.random.choice(a=alphabet_loc, size=2, replace=False)
                    alphabet_loc.remove(allow[0])
                    alphabet_loc.remove(allow[1])

                    # 多出来的字母，替换为数字
                    new_nos = np.random.choice(a=LicensePlateNoGenerator.numerals, size=len(alphabet_loc), replace=True)
                    for j, loc in enumerate(alphabet_loc):
                        lic_no[loc] = new_nos[j]
        else:
            for char_index in range(7):
                # 对应车牌位上限制的字符范围
                char_range = self.elements.get_chars_sorted_by_label(char_index)
                if char_index == 0:
                    char_range = char_range[:31]  # 第一位排除掉‘军’和‘使’
                elif char_index == 1:
                    # 第二位的范围还和省份相关，这里没考虑
                    char_range = char_range[:24]  # 第二位排除数字
                elif char_index == 6:
                    char_range = char_range[:34]  # 第六位排除‘学’、‘警’等特殊字符

                license_chars.append(np.random.choice(a=char_range, size=num, replace=True))

            # 取每一位，组成7位车牌
            license_plate_numbers = [list(_) for _ in zip(*license_chars)]
            # 在后五位编码可以出现字母，但不能超过两个
            for i, lic_no in enumerate(license_plate_numbers):
                # 找出后5位中英文字母的位置
                alphabet_loc = list()
                for loc in range(2, 7):
                    if lic_no[loc] in LicensePlateNoGenerator.alphabet:
                        alphabet_loc.append(loc)
                # 字母数多于两个的，随机保留2个（这样会导致车牌中2位字母数的车牌比较多）
                if len(alphabet_loc) > 2:
                    allow = np.random.choice(a=alphabet_loc, size=2, replace=False)
                    alphabet_loc.remove(allow[0])
                    alphabet_loc.remove(allow[1])

                    # 多出来的字母，替换为数字
                    new_nos = np.random.choice(a=LicensePlateNoGenerator.numerals, size=len(alphabet_loc), replace=True)
                    for j, loc in enumerate(alphabet_loc):
                        lic_no[loc] = new_nos[j]

        license_plate_numbers = [''.join(_) for _ in license_plate_numbers]
        return license_plate_numbers

    def generate_jun_license_no(self, num):
        license_char = list()
        first_char = ['V', 'K', 'H', 'B', 'S', 'L', 'J', 'N', 'G', 'C']
        second_char = ['A', 'B', 'C', 'D', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V', 'Y']
        for i in range(7):
            if i == 0:
                char = first_char
            elif i == 1:
                char = second_char
            else:
                char = digits
            license_char.append(np.random.choice(char, num, replace=True))
        license_plate_numbers = [list(_) for _ in zip(*license_char)]
        license_plate_numbers = [''.join(_) for _ in license_plate_numbers]
        return license_plate_numbers

    def generate_WJ_license_no(self, num, header=True):
        plate_char = list()
        last_char = ['J','B', 'X', 'T', 'H', 'S', 'D']
        two_char = "WJ"
        if header:
            for i in range(5):
                if i == 4:
                    char = last_char + digits
                else:
                    char = digits
                plate_char.append(np.random.choice(char, num, replace=True))
        else:
            for i in range(6):
                if i == 0:
                    char = provinces
                elif i == 5:
                    char = last_char + digits
                else:
                    char = digits
                plate_char.append(np.random.choice(char, num, replace=True))
        license_plate_numbers = [list(_) for _ in zip(*plate_char)]
        license_plate_numbers = [two_char + ''.join(_) for _ in license_plate_numbers]
        return license_plate_numbers

    def generate_gangao_license_no(self, num, plate_type):
        plate_char = list()
        for i in range(4):
            char = digits
            plate_char.append(np.random.choice(char, num, replace=True))
        license_plate_numbers = [list(_) for _ in zip(*plate_char)]
        if plate_type == 'hk':
            license_plate_numbers = ['NS'+''.join(_) for _ in license_plate_numbers]
        else:
            license_plate_numbers = ['MC'+''.join(_) for _ in license_plate_numbers]
        return license_plate_numbers

    def generate_black_gangao_license_no(self, num, plate_type):
        plate_char = list()
        two_char = '粤Z'
        for i in range(4):
            char = digits + letters
            plate_char.append(np.random.choice(char, num, replace=True))
        license_plate_numbers = [list(_) for _ in zip(*plate_char)]
        for i, lic_no in enumerate(license_plate_numbers):
                # 找出后5位中英文字母的位置
                alphabet_loc = list()
                for loc in range(0, 4):
                    if lic_no[loc] in LicensePlateNoGenerator.alphabet:
                        alphabet_loc.append(loc)
                # 字母数多于两个的，随机保留2个（这样会导致车牌中2位字母数的车牌比较多）
                if len(alphabet_loc) > 2:
                    allow = np.random.choice(a=alphabet_loc, size=2, replace=False)
                    alphabet_loc.remove(allow[0])
                    alphabet_loc.remove(allow[1])

                    # 多出来的字母，替换为数字
                    new_nos = np.random.choice(a=LicensePlateNoGenerator.numerals, size=len(alphabet_loc), replace=True)
                    for j, loc in enumerate(alphabet_loc):
                        lic_no[loc] = new_nos[j]
        if plate_type == 'black_gang':
            license_plate_numbers = [two_char + ''.join(_) for _ in license_plate_numbers]
            license_plate_numbers = [i+'港' for i in license_plate_numbers]
        else:
            license_plate_numbers = [two_char + ''.join(_) for _ in license_plate_numbers]
            license_plate_numbers = [i+'澳' for i in license_plate_numbers]

        return license_plate_numbers

    def generate_embra_license_no(self, num, plate_type):
        plate_char = list()
        first_char = '使'
        last_char = '领'
        if 'motor_shi' == plate_type:
            two_char = first_char + 'A'
            for i in range(4):
                char = digits
                plate_char.append(np.random.choice(char,num,replace=True))
            license_plate_numbers = [list(_) for _ in zip(*plate_char)]
            license_plate_numbers = [two_char+''.join(_) for _ in license_plate_numbers]
        else:
            if 'shi' in plate_type:
                three_char = ['{}'.format(x + 1) for x in range(100, 318)]
                for i in range(4):
                    if i == 0:
                        char = three_char
                    else:
                        char = digits
                    plate_char.append(np.random.choice(char,num,replace=True))
                license_plate_numbers = [list(_) for _ in zip(*plate_char)]
                license_plate_numbers = [first_char+''.join(_) for _ in license_plate_numbers]
            else:
                for i in range(6):
                    if i == 0:
                        char = provinces
                    elif i == 1:
                        char = ['A']
                    else:
                        char = digits
                    plate_char.append(np.random.choice(char, num, replace=True))
                license_plate_numbers = [list(_) for _ in zip(*plate_char)]
                license_plate_numbers = [''.join(_) for _ in license_plate_numbers]
                license_plate_numbers = [i+last_char for i in license_plate_numbers]
        return license_plate_numbers

    def generate_old_license_no(self, num, plate_type):
        plate_char = list()
        last_char1 = '领'
        last_char2 = '使'
        if 'old_shi' == plate_type:
            three_char = ['{}'.format(x + 1) for x in range(100, 318)]
            for i in range(4):
                if i == 0:
                    char = three_char
                else:
                    char = digits
                plate_char.append(np.random.choice(char, num, replace=True))
            license_plate_numbers = [list(_) for _ in zip(*plate_char)]
            license_plate_numbers = [''.join(_) for _ in license_plate_numbers]
            license_plate_numbers = [i + last_char2 for i in license_plate_numbers]
        else:
            for i in range(6):
                if i == 0:
                    char = provinces
                else:
                    char = digits
                plate_char.append(np.random.choice(char, num, replace=True))
            license_plate_numbers = [list(_) for _ in zip(*plate_char)]
            license_plate_numbers = [''.join(_) for _ in license_plate_numbers]
            license_plate_numbers = [i + last_char1 for i in license_plate_numbers]
        return license_plate_numbers









