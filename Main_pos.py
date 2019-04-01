import numpy as nmp
import tensorflow as tf
import cv2 as cv


PATH_ORIG = 'C:/TF/Det_rec/Orig/'
PATH_SAVE_NET = 'C:/TF/Det_rec/Train_res/v1/'
PATH_SAVE_NET_PARTS = 'C:/TF/Det_rec/Train_res/v2/'
PATH_SAVE_PARTS_DATA = 'C:/TF/Det_rec/Parts/'
PATH_LOAD_PARTS_DATA_IMAGE = 'C:/TF/Det_rec/Parts/Lines/'


def all_together_3():
    """Объединение классификатора и позиционирования"""
    # cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('C:/TF/Det_rec/For_save_img/output.avi', fourcc, 20.0, (640, 480))
    cap = cv.VideoCapture('C:/TF/Det_rec/For_save_img/input.avi')

    eng_text = {
        0: 'BackGround',
        1: 'TELE2',
        2: 'MAXIDOM',
        3: 'MOLLIES',
        4: 'OHOOLY',
        5: 'BLACK SQR',
        6: 'DET_SPLASH',
        7: 'DET_NUMBER',
    }

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
    pos_class = MainPositionClass(prep_image, [290, 100, (480, 640), 10])

    det_recog = NNDR(train=False,
                     inp_image_sz=(480, 640),
                     n_of_classes=8,
                     scale_size=(100, 100))

    last_frame = nmp.zeros([480, 640, 3], nmp.uint8)
    pre_last_frame = nmp.zeros([480, 640, 3], nmp.uint8)
    font = cv.FONT_HERSHEY_SIMPLEX

    sv = tf.Session()
    print("Сессия готова...")
    with sv as sess:
        sess.run(tf.global_variables_initializer())

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'NNDR')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, 'C:/TF/Det_rec/Train_res/v1/ckp')
        print("1 Restored!")

        while True:
            e1 = cv.getTickCount()

            ret, frame = cap.read()
            if ret:
                fr1 = frame
                # fr1 = cv.GaussianBlur(fr1, (5, 5), 0.5)
                # fr1 = cv.add(last_frame, fr1) // 2
                fr1 = ((fr1.astype(nmp.float32) + last_frame.astype(nmp.float32)) / 2).astype(nmp.uint8)
                # fr1 = ((fr1.astype(nmp.float32)
                #         + last_frame.astype(nmp.float32)
                #         + pre_last_frame.astype(nmp.float32)) / 3).astype(nmp.uint8)
                # fr1 = cv.cvtColor(fr1, cv.COLOR_RGB2BGR)
                # pre_last_frame = last_frame
                last_frame = frame

                if nmp.mean(fr1) < 130:
                    k = 130 / nmp.mean(fr1)
                    fr1 = nmp.clip(fr1 * k, 0, 255).astype(nmp.uint8)
                if nmp.mean(fr1) > 200:
                    k = 200 / nmp.mean(fr1)
                    fr1 = nmp.clip(fr1 * k, 0, 255).astype(nmp.uint8)

                cv.putText(fr1, str(nmp.mean(fr1)),
                           (10 * 1, 400 * 1),
                           font, 0.5 * 1, (0, 255, 0), 1 * 1, cv.LINE_AA)

                # Первый шаг
                returned_frame_set = pos_class.first_step(fr1)
                if returned_frame_set is None:
                    scale_factor = fr1.shape[0] // 480
                    cv.putText(fr1, 'No details',
                               (10 * scale_factor, 20 * scale_factor),
                               font, 0.5 * scale_factor, (255, 0, 0), 1 * scale_factor, cv.LINE_AA)
                    # time.sleep(0.07)
                    mini_frame = cv.resize(fr1, (800, 600))
                    cv.imshow('img_orig', mini_frame)
                    out.write(fr1)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                frame_det = nmp.array(returned_frame_set)

                ans_arr_detail = sess.run([det_recog.last_layer],
                                          feed_dict={det_recog.input_batch_RGB_ph_pp: frame_det})

                # Второй шаг
                classify_result_list = []
                for each_detail in ans_arr_detail[0]:
                    classify_result_list.append(nmp.argmax(each_detail))

                coord_list, finding_lines = pos_class.second_step(classify_result_list)

                # Третий шаг
                names_list = []
                for each_detail in classify_result_list:
                    names_list.append(eng_text[int(each_detail)])
                res_image = pos_class.third_step(coord_list, finding_lines, names_list)

                mini_frame = cv.resize(res_image, (800, 600))
                cv.imshow('img_orig', mini_frame)
                out.write(res_image)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                e2 = cv.getTickCount()
                compute_time = (e2 - e1) / cv.getTickFrequency() * 1000
                print('Цикл: {0:.2f} ms'.format(compute_time))

            else:
                print('Пропущен кадр')
                break
                # if cv.waitKey(1) & 0xFF == ord('q'):
                #     break

    cv.destroyAllWindows()
    cap.release()
    out.release()