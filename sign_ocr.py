# coding=utf-8
import cv2
import sys
import os
import time
import common as common
import platform

reload(sys)                      # reload 才能调用 setdefaultencoding 方法
sys.setdefaultencoding('utf-8')  # 设置 'utf-8'


class SignOcr:
    def __init__(self, image_dir):
        self.img = None
        self.img_files = common.get_files(image_dir)
        print('total imgs len: ', len(self.img_files))
        self.image_dir = image_dir
        self.plate_encode = "utf8"
        self.label_normal_file = os.path.join('.', 'label_normal.txt')
        self.label_test_file = os.path.join('.', 'label_test.txt')
        self.label_error_file = os.path.join('.', 'label_error.txt')
        self.index_file = os.path.join('.', 'index.txt')

        return

    def use_platform(self):
        sysstr = platform.system()
        if sysstr == "Windows":
            self.plate_encode = "gb2312"
            print ("current system: windows")
        elif sysstr == "Linux":
            self.plate_encode = "utf8"
            print ("current system: linux")
        else:
            print ("Other System ")

    def review_start(self, num=12):
        print('[review_start] ...')
        times = 2
        label_list = []

        with open(self.label_normal_file, 'r') as f:
            for line in f.readlines():
                lines = line.replace("\n", "").split(":")
                label_list.append([lines[0], lines[1]])

        with open(self.label_green_file, 'r') as f:
            for line in f.readlines():
                lines = line.replace("\n", "").split(":")
                label_list.append([lines[0], lines[1]])

        # print(len(label_list))
        # print((len(label_list)+num-1) / num)
        for i in range((len(label_list)+num-1) / num):
            for j in range(num):
                img = cv2.imread(label_list[i*num+j][0])
                img = cv2.resize(img, (img.shape[1]*times, img.shape[0]*times))

                # plate_name = label_list[i * num + j][1]                                           # utf8
                plate_name = label_list[i * num + j][1].decode(self.plate_encode)                   # unicode
                # plate_name = label_list[i * num + j][1].decode(self.plate_encode).encode('gbk')   # gbk

                print(plate_name)
                print(type(plate_name))
                print('111', plate_name)
                cv2.imshow(plate_name, img)
                # cv2.imshow(label_list[i * num + j][1].decode('utf-8'), img)
                # cv2.moveWindow(label_list[i*num+j][1].decode('gbk'), 400 * (j / 4) + 100, 200 * (j % 4) + 50)
            cv2.waitKey(0)
            # cv2.destroyWindow('image')
            # str = raw_input('wait ...')
        return

    def save_label(self, file_name, plate, times=1):
        print('save_label ...')
        data = file_name + ":" + plate.encode(self.plate_encode) + '\n'
        print('[save_label] plate len: %d' % len(plate))

        if len(plate) == 7:  # 正常车牌
            if times == 1:
                common.write_data(self.label_normal_file, data, 'a+')
            else:
                for i in range(times):
                    common.write_data(self.label_normal_file, data, 'a+')
                common.write_data(self.label_test_file, data, 'a+')
        elif len(plate) == 8:  # 新能源车牌
            if times == 1:
                common.write_data(self.label_normal_file, data, 'a+')
            else:
                for i in range(times):
                    common.write_data(self.label_normal_file, data, 'a+')
                common.write_data(self.label_test_file, data, 'a+')
        else:  # 其他车牌
            common.write_data(self.label_error_file, data, 'a+')

    def sign_start(self, restart=False):
        times = 4               # 图片放大倍数

        if restart is False:
            try:
                start_i = int(common.read_data(self.index_file, 'r'))
                print('start_index: ' + str(start_i))
            except Exception, e:
                print e
                start_i = 0
        else:
            start_i = 0

        while start_i < len(self.img_files):
            print('[total] %d; [index] %d; [name] %s' % (len(self.img_files), start_i, self.img_files[start_i]))
            plate = self.img_files[start_i].split(os.sep)[-1].split('_')[1].split('.')[0]
            # plate = plate.decode('utf8')
            plate = plate.decode(self.plate_encode)
            print('[plate] %s' % plate)

            # print(self.img_files[start_i])
            # print(type(self.img_files[start_i]))
            self.img = cv2.imread(self.img_files[start_i])
            self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0]*times))
            cv2.imshow('sign_image', self.img)

            while True:
                cv2.imshow('sign_image', self.img)

                # 保存这张图片
                k = cv2.waitKey(1) & 0xFF
                if k == ord('s'):
                    self.save_label(self.img_files[start_i], plate)
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

                if k == ord('1'):
                    self.save_label(self.img_files[start_i], plate, 1)
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

                if k == ord('2'):
                    self.save_label(self.img_files[start_i], plate, 2)
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

                if k == ord('3'):
                    self.save_label(self.img_files[start_i], plate, 3)
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

                if k == ord('5'):
                    self.save_label(self.img_files[start_i], plate, 5)
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

                if k == ord('d'):
                    print('delete ...')
                    common.exe_cmd('rm -r ' + self.img_files[start_i])
                    self.img_files.pop(start_i)
                    self.img = cv2.imread(self.img_files[start_i])
                    self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0]*times))
                    cv2.imshow('sign_image', self.img)
                    break

                if k == ord('c'):
                    plate = raw_input('input new plate: ')
                    plate = plate.decode(self.plate_encode)
                    count = raw_input('input save time: ')
                    self.save_label(self.img_files[start_i], plate, int(count))
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

    def check_start(self, root_path, label_name, restart=False):
        if restart is False:
            try:
                start_i = int(common.read_data(self.index_file, 'r'))
                print('start_index: ' + str(start_i))
            except Exception, e:
                print e
                start_i = 0
        else:
            start_i = 0

        index = 0
        times = 4

        with open(os.path.join(root_path, label_name)) as f:
            label_lines = f.readlines()
            total_len = len(label_lines)
            for line in label_lines:
                if index >= start_i:
                    line = line.replace('\r', '').replace('\n', '')
                    print(line)
                    list_str = line.split(':')
                    print("[%d/%d] %s" % (index, total_len, list_str[1]))
                    self.img = cv2.imread(os.path.join(root_path, list_str[0]))
                    self.img = cv2.resize(self.img, (self.img.shape[1] * times, self.img.shape[0] * times))

                    while True:
                        cv2.imshow('check_image', self.img)

                        # 保存这张图片
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('y'):
                            common.write_data('error.txt', line + "\n", 'a+')
                            break

                        if k == ord('n'):
                            break

                    index += 1
                    start_i = index
                    common.write_data('index.txt', str(start_i), 'w')
                else:
                    index += 1


if __name__ == '__main__':
    # image_dir = "../Data/car_recognition/train/province_2"
    # image_dir = "../Data/car_recognition/train/province_3"
    # image_dir = "../Data/car_recognition/train/province_4"
    # image_dir = "../Data/car_recognition/train/failed_5"
    # image_dir = "../Data/car_recognition/train/failed_7"
    # image_dir = "../Data/car_recognition/train/failed_9"
    image_dir = "../Data/car_recognition/train/failed_11"

    sign_ocr = SignOcr(image_dir)
    sign_ocr.use_platform()

    sign_ocr.sign_start()
    # sign_ocr.check_start("../Data/car_recognition/train", "labels_normal.txt")
    # sign_ocr.review_start()

