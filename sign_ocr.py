# coding=utf-8
import cv2
import sys
import os
import time
import common as common

# ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON',
# 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK',
# 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

# events = [i for i in dir(cv2) if 'EVENT' in i]
# img = np.zeros((512, 512, 3), np.uint8)


# mouse callback function
# def mouse_click_events(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

reload(sys)                      # reload 才能调用 setdefaultencoding 方法
sys.setdefaultencoding('utf-8')  # 设置 'utf-8'

class SignOcr:
    def __init__(self, image_dir):
        self.img = None
        self.img_files = common.get_files(image_dir)
        self.image_dir = image_dir
        # self.car_points = []
        self.label_normal_file = './label_normal.txt'
        self.label_green_file = './label_green.txt'
        self.label_error_file = './label_error.txt'
        self.index_file = './index.txt'
        return

    # def mouse_click_events(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         if len(self.car_points) < 4:
    #             cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
    #             print('click: [%d, %d]' % (x, y))
    #             self.car_points.append((x, y))
    #         else:
    #             print('self.car_points is too long, %s' % str(self.car_points))

    def save_label(self, file_name, plate):
        print('save_label ...')
        data = file_name + ":" + plate + '\n'
        print('[plate] %d' % len(plate))

        if len(plate) == 7:  # 正常车牌
            common.write_data(self.label_normal_file, data, 'a+')
        elif len(plate) == 8:  # 新能源车牌
            common.write_data(self.label_green_file, data, 'a+')
        else:  # 其他车牌
            common.write_data(self.label_error_file, data, 'a+')

    def sign_start(self, restart=False):
        times = 4

        # cv2.namedWindow('sign_image')
        # cv2.setMouseCallback('sign_image', self.mouse_click_events)    # 鼠标事件绑定

        if restart is False:
            try:
                start_i = int(common.read_data(self.index_file, 'r'))
                print('start_index: ' + str(start_i))
            except Exception, e:
                print e
                start_i = 0
        else:
            start_i = 0

        # for img_file in self.img_files:
        while start_i < len(self.img_files):
            print('[total] %d; [index] %d; [name] %s' % (len(self.img_files), start_i, self.img_files[start_i]))
            plate = self.img_files[start_i].split('/')[-1].split('_')[1].split('.')[0]
            plate = plate.decode('utf8')
            print('[plate] %s' % plate)

            self.img = cv2.imread(self.img_files[start_i])
            self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0]*times))
            cv2.imshow('sign_image', self.img)

            while True:
                cv2.imshow('sign_image', self.img)

                # 保存这张图片
                k = cv2.waitKey(1) & 0xFF
                if k == ord('s'):
                    self.save_label(self.img_files[start_i], plate)
                    # data = self.img_files[start_i] + ":" + plate + '\n'
                    #
                    # if len(plate) == 7:     # 正常车牌
                    #     common.write_data(self.label_normal_file, data, 'a+')
                    # elif len(plate) == 8:   # 新能源车牌
                    #     common.write_data(self.label_green_file, data, 'a+')
                    # else:                   # 其他车牌
                    #     common.write_data(self.label_error_file, data, 'a+')

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
                    plate = plate.decode('utf8')
                    self.save_label(self.img_files[start_i], plate)
                    # if len(plate) == 7:     # 正常车牌
                    #     common.write_data(self.label_normal_file, data, 'a+')
                    # elif len(plate) == 8:   # 新能源车牌
                    #     common.write_data(self.label_green_file, data, 'a+')
                    # else:                   # 其他车牌
                    #     common.write_data(self.label_error_file, data, 'a+')

                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    break

                # 重新加载图片
                # if k == ord('r'):
                #     print('re sign ...')
                #     self.img = cv2.imread(self.img_files[start_i])
                #     self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0]*times))
                #     cv2.imshow('sign_image', self.img)
                #     self.car_points = []

                # 改变图片大小
                # if k == ord('c'):
                #     print('change size ...')
                #     if times == 2:
                #         times = 4
                #     else:
                #         times = 2
                #     self.img = cv2.imread(self.img_files[start_i])
                #     self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0]*times))
                #     cv2.imshow('sign_image', self.img)
                #     self.car_points = []


if __name__ == '__main__':
    # image_dir = "../Data/car_recognition/test/blue_2"
    # image_dir = "../Data/car_recognition/test/blue_3"
    image_dir = "../Data/car_recognition/test/blue_failed_1"

    # image_dir = "../Data/car_recognition/train/blue_闽_1"
    # image_dir = "../Data/car_recognition/train/blue_yue_1"
    # image_dir = "../Data/car_recognition/train/blue_粤_1"
    # image_dir = "../Data/car_recognition/train/blue_鄂_1"
    # image_dir = "../Data/car_recognition/train/blue_京_1"
    # image_dir = "../Data/car_recognition/train/blue_苏_1"
    # image_dir = "../Data/car_recognition/train/blue_浙_1"
    # image_dir = "../Data/car_recognition/train/blue_failed_1"

    # label_file = "./label.txt"
    # index_file = "./index.txt"
    sign_ocr = SignOcr(image_dir)

    sign_ocr.sign_start()
