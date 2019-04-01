import numpy as nmp
import tensorflow as tf
import cv2 as cv
from tqdm import tqdm
from datetime import datetime
from Positioning_system.Network_arch import NNDR


PATH_ORIG = 'C:/TF/Det_rec/Orig/'
PATH_SAVE_NET = 'C:/TF/Det_rec/Train_res/v1/'
PATH_SAVE_NET_PARTS = 'C:/TF/Det_rec/Train_res/v2/'
PATH_SAVE_PARTS_DATA = 'C:/TF/Det_rec/Parts/'
PATH_LOAD_PARTS_DATA_IMAGE = 'C:/TF/Det_rec/Parts/Lines/'


def train_img_generator(list_of_images: list, size_of_batch: int, add_noise=False, rnd_brightness=True):
    """Генерация изображений н лету из оригиналов. Положение в списке должно соответствовать классам"""
    background = list_of_images[0]
    background = cv.resize(background, (640, 480), interpolation=cv.INTER_NEAREST)
    filling_color = nmp.mean(nmp.mean(background, axis=0), axis=0)
    loc_list_of_images = list_of_images[1:]
    image_h, image_w = background.shape[:2]
    image_h_half, image_w_half = image_h // 2, image_w // 2

    size_of_one_class = size_of_batch // len(list_of_images)
    total_elems = size_of_one_class * len(list_of_images)
    image_array = nmp.zeros([total_elems, image_h, image_w, 3], dtype=nmp.uint8)
    class_array = nmp.zeros([total_elems], dtype=nmp.uint8)

    random_angles = nmp.random.rand(len(loc_list_of_images), size_of_one_class) * 360
    random_coords = nmp.random.randint(-10, 10, [len(loc_list_of_images), size_of_one_class, 2])
    random_scales = (nmp.random.rand(len(loc_list_of_images), size_of_one_class) - 0.5) / 5 + 1
    random_brightness = (nmp.random.rand(len(loc_list_of_images), size_of_one_class) - 0.5) * 2 * 0.40 + 1

    for num, image in enumerate(loc_list_of_images):
        class_array[size_of_one_class * num: size_of_one_class * (num + 1)] = num + 1

        canny_image = cv.Canny(image, 100, 200)
        contour = cv.findContours(canny_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

        new_image = nmp.ones_like(background)
        new_image = (new_image * filling_color).astype(nmp.uint8)

        contour_mask = nmp.zeros([image_h, image_w])
        contour_mask = cv.fillPoly(contour_mask, contour, 1).astype(nmp.bool)

        (x_center, y_center), rad = cv.minEnclosingCircle(contour[0])
        x_roll_mask, y_roll_mask = x_center - new_image.shape[1] // 2, y_center - new_image.shape[0] // 2
        x_roll_mask, y_roll_mask = int(x_roll_mask), int(y_roll_mask)
        dest_mask = nmp.roll(contour_mask, (-x_roll_mask, -y_roll_mask), (1, 0)).astype(nmp.bool)

        new_image[dest_mask] = image[contour_mask]

        square_coords = nmp.array([[[image_w_half + rad, image_h_half - rad]],
                                   [[image_w_half - rad, image_h_half - rad]],
                                   [[image_w_half - rad, image_h_half + rad]],
                                   [[image_w_half + rad, image_h_half + rad]]]).astype(nmp.int32)
        square_mask = nmp.zeros([image_h, image_w])
        square_mask = cv.fillPoly(square_mask, [square_coords], 1).astype(nmp.bool)

        for gen in range(size_of_one_class):
            angle_image = nmp.ones([image_h, image_w, 3])
            angle_image = (angle_image * filling_color).astype(nmp.uint8)

            rot_mat = cv.getRotationMatrix2D((image_w_half, image_h_half),
                                             random_angles[num, gen], random_scales[num, gen])

            rot_image = cv.warpAffine(new_image, rot_mat, (image_w, image_h))

            angle_image[square_mask] = rot_image[square_mask]

            shift_image = nmp.roll(angle_image, random_coords[num, gen], [0, 1])

            if add_noise:
                noise_param = int(2 * nmp.random.rand(1)) + 2
                shift_image = nmp.clip(shift_image + nmp.random.randint(-noise_param, noise_param,
                                                                        [image_h, image_w, 3]),
                                       0, 255).astype(nmp.uint8)

            if rnd_brightness:
                shift_image = nmp.clip(shift_image * random_brightness[num, gen], 0, 255).astype(nmp.uint8)

            shift_image = cv.resize(shift_image, (640, 480), interpolation=cv.INTER_NEAREST)

            image_array[num * size_of_one_class + gen] = shift_image

    class_array[-size_of_one_class:] = 0

    noise_array = nmp.random.randint(-5, 5, [size_of_one_class, image_h, image_w, 3])
    image_array[-size_of_one_class:] = nmp.clip((background + noise_array), 0, 255).astype(nmp.uint8)

    rnd_state = nmp.random.get_state()
    nmp.random.shuffle(image_array)
    nmp.random.set_state(rnd_state)
    nmp.random.shuffle(class_array)

    return image_array, class_array


def get_log_directory():
    """Создаёт путь для логов с учётом времени начала работы"""
    right_now_time = datetime.now()
    logdir = '{0}_{1}_{2} ({3}-{4})__'.format(right_now_time.day,
                                              right_now_time.month,
                                              right_now_time.year,
                                              right_now_time.hour,
                                              right_now_time.minute)
    return logdir


def main_train_f():
    """Обучение свёрточной нейронной сети на генерируемой выборке"""

    backgr = cv.imread(PATH_ORIG + '0_1.png')
    im_cl_1 = cv.imread(PATH_ORIG + '1_1.png')
    im_cl_2 = cv.imread(PATH_ORIG + '2_1.png')
    im_cl_3 = cv.imread(PATH_ORIG + '3_1.png')
    im_cl_4 = cv.imread(PATH_ORIG + '4_1.png')
    im_cl_5 = cv.imread(PATH_ORIG + '5_1.png')
    im_cl_6 = cv.imread(PATH_ORIG + '6_1.png')
    im_cl_7 = cv.imread(PATH_ORIG + '7_1.png')
    prep_image = [backgr, im_cl_1, im_cl_2, im_cl_3, im_cl_4,
                  im_cl_5, im_cl_6, im_cl_7]
    image_pack = []
    for next_image in prep_image:
        image_pack.append(cv.cvtColor(next_image, cv.COLOR_BGR2RGB))
    imgs, classes = train_img_generator(image_pack, 500)

    test_imgs, test_classes = train_img_generator(image_pack, 20)

    det_recog = NNDR(train=True,
                     inp_image_sz=imgs.shape[1:3],
                     n_of_classes=8,
                     scale_size=(100, 100),
                     learn_rate=2E-3)
    print("Граф операций загружен!")

    sv = tf.Session()
    with sv as sess:
        summ = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        print('Начинаю загрузку датасета...')
        sess.run(det_recog.iterate.initializer, feed_dict={det_recog.input_batch_RGB_ph_pp: imgs,
                                                           det_recog.output_batch_ph: classes})
        print('Загрузка завершена!')

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('C:/TF/Log/' + 'Vars' + get_log_directory())
        test_writer = tf.summary.FileWriter('C:/TF/Log/' + 'Vars_test' + get_log_directory())
        writer.add_graph(sess.graph)

        limit = 6000
        gs = 0
        while gs < limit - 1:
            for _ in tqdm(range(limit),
                          total=limit,
                          ncols=100,
                          leave=False,
                          unit='b'):
                gs, _ = sess.run([det_recog.global_step, det_recog.optim])
                if gs % 100 == 0:
                    s = sess.run(summ)
                    writer.add_summary(s, gs)
                    saver.save(sess, PATH_SAVE_NET + '/ckp')

                if gs % 100 == 0:
                    test_imgs, test_classes = train_img_generator(image_pack, 20, True)

                    rnd_state = nmp.random.get_state()
                    nmp.random.shuffle(test_imgs)
                    nmp.random.set_state(rnd_state)
                    nmp.random.shuffle(test_classes)

                    sess.run(det_recog.iterate.initializer, feed_dict={det_recog.input_batch_RGB_ph_pp: test_imgs,
                                                                       det_recog.output_batch_ph: test_classes})

                    s = sess.run(summ)
                    test_writer.add_summary(s, gs)

                    imgs, classes = train_img_generator(image_pack, 1000, True)

                    sess.run(det_recog.iterate.initializer, feed_dict={det_recog.input_batch_RGB_ph_pp: imgs,
                                                                       det_recog.output_batch_ph: classes})

        s = sess.run(summ)
        writer.add_summary(s, gs)
        saver.save(sess, PATH_SAVE_NET + '/ckp')

        rnd_state = nmp.random.get_state()
        nmp.random.shuffle(test_imgs)
        nmp.random.set_state(rnd_state)
        nmp.random.shuffle(test_classes)

        sess.run(det_recog.iterate.initializer, feed_dict={det_recog.input_batch_RGB_ph_pp: test_imgs,
                                                           det_recog.output_batch_ph: test_classes})
        s = sess.run(summ)
        test_writer.add_summary(s, gs)

    print("Done")
