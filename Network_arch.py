import tensorflow as tf


class NNDR:
    """Сеть для распознавания отдельных классов деталей"""

    def __init__(self,
                 train=True,
                 inp_image_sz=(480, 640),
                 n_of_classes=2,
                 scale_size=(100, 100),
                 learn_rate=1E-5):

        with tf.variable_scope('NNDR'):
            self.input_batch_RGB_ph_pp = tf.placeholder(tf.uint8,
                                                        [None, inp_image_sz[0], inp_image_sz[1], 3])

            self.input_batch_RGB_ph_ps = tf.image.resize_bicubic(self.input_batch_RGB_ph_pp, (200, 200))

            if train:
                """Создание датасета из полученных изображений"""
                self.output_batch_ph = tf.placeholder(tf.int32, [None])
                self.output_batch = tf.one_hot(self.output_batch_ph, n_of_classes)
                # self.output_batch_ph = self.output_batch_ph[:, tf.newaxis, tf.newaxis, :]

                self.dataset = tf.data.Dataset.from_tensor_slices((self.input_batch_RGB_ph_ps,
                                                                   self.output_batch)).repeat().batch(20)
                self.iterate = self.dataset.make_initializable_iterator()
                self.input_batch_s, self.output_batch_s = self.iterate.get_next()
                self.output_batch = tf.reshape(self.output_batch_s, [-1, n_of_classes])

                """Преобразование изображений и получение градиента"""
                self.input_batch_RGB_ph = tf.image.random_brightness(self.input_batch_s, 0.5)
                self.input_batch_RGB_ph = tf.image.random_hue(self.input_batch_RGB_ph, 0.08)
                self.input_batch_RGB_ph = tf.image.random_saturation(self.input_batch_RGB_ph, 0.8, 1.2)
                # self.input_batch_RGB_ph = tf.image.random_flip_up_down(self.input_batch_RGB_ph)
                # self.input_batch_RGB_ph = tf.image.random_flip_left_right(self.input_batch_RGB_ph)
                self.input_batch_RGB_ph = tf.image.random_contrast(self.input_batch_RGB_ph, 0.8, 1.2)

                self.grad_of_images = tf.image.rgb_to_grayscale(self.input_batch_RGB_ph)
                self.grad_of_images = self.real_time_grad(self.grad_of_images,
                                                          stream=False)
                # self.grad_of_images = tf.concat((self.grad_of_images,
                #                                  self.grad_of_images,
                #                                  self.grad_of_images), axis=3)

                self.grad_of_images = tf.image.resize_bicubic(self.grad_of_images, scale_size)  # [:, :, :, :1]
                self.input_batch_RGB_ph = tf.image.resize_bicubic(self.input_batch_RGB_ph, scale_size)

                """Нормализация и зашумление"""
                self.input_batch = tf.concat((self.input_batch_RGB_ph, self.grad_of_images),
                                             axis=3)
                self.input_batch = tf.divide(self.input_batch, 255)
                self.input_batch = tf.layers.dropout(self.input_batch, 0.15)

                tf.summary.image('Origs', self.input_batch[:, :, :, :3])
                tf.summary.image('grads', self.input_batch[:, :, :, 3:4])
            else:
                """Расчёт градиента"""
                self.grad_of_images = tf.image.rgb_to_grayscale(self.input_batch_RGB_ph_pp)
                self.grad_of_images = self.real_time_grad(self.grad_of_images,
                                                          stream=False)
                # self.grad_of_images = tf.concat((self.grad_of_images,
                #                                  self.grad_of_images,
                #                                  self.grad_of_images), axis=3)

                self.grad_of_images = tf.image.resize_bicubic(self.grad_of_images, scale_size)  # [:, :, :, :1]
                self.input_batch_RGB_ph = tf.image.resize_bicubic(self.input_batch_RGB_ph_pp, scale_size)

                """Нормализация"""
                self.input_batch = tf.concat((self.input_batch_RGB_ph, self.grad_of_images),
                                             axis=3)
                self.input_batch = tf.divide(self.input_batch, 255)

            self.lay_1 = tf.layers.conv2d(inputs=self.input_batch,
                                          filters=8,
                                          kernel_size=[10, 10],
                                          strides=[1, 1],
                                          activation=tf.nn.relu)

            self.lay_1_mp = tf.layers.max_pooling2d(inputs=self.lay_1,
                                                    pool_size=2,
                                                    strides=1)

            self.lay_2 = tf.layers.conv2d(inputs=self.lay_1_mp,
                                          filters=16,
                                          kernel_size=[2, 2],
                                          strides=[1, 1],
                                          activation=tf.nn.relu)

            self.lay_3_pr = tf.layers.flatten(self.lay_2)

            if train:
                self.lay_3_pr = tf.layers.dropout(self.lay_3_pr, 0.15)

            self.lay_3 = tf.layers.dense(inputs=self.lay_3_pr,
                                         units=128,
                                         activation=tf.nn.sigmoid)

            self.lay_4 = tf.layers.dense(inputs=self.lay_3,
                                         units=n_of_classes,
                                         activation=None)

            self.last_layer = self.lay_4

            if train:
                with tf.variable_scope("gs"):
                    self.global_step = tf.train.get_or_create_global_step()

                # self.xent = -tf.reduce_sum(self.output_batch * tf.log(self.last_layer), name="xent")
                self.xent = tf.reduce_max(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.last_layer,
                                                                                     labels=self.output_batch))
                tf.summary.scalar("xent_1", self.xent)

                self.correct_prediction = tf.equal(tf.argmax(self.last_layer, 1), tf.argmax(self.output_batch, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

                # self.loc_lear = learn_rate / tf.math.log1p(((self.global_step + 1) / 100) + 1)
                self.loc_lear = learn_rate * tf.pow(tf.cast(0.1, tf.float64), self.global_step / 3000 + 1)
                # self.loc_lear = learn_rate / 10
                self.optim = tf.train.AdamOptimizer(learning_rate=self.loc_lear).minimize(self.xent,
                                                                                          global_step=self.global_step)
                tf.summary.scalar("learning_rate", self.loc_lear)

    @staticmethod
    def real_time_grad(inp_image: tf.Tensor, stream=True):
        """Метод для расчёта градиента изображений композитным способом"""
        # grad
        if stream:
            mem = tf.Variable(tf.zeros(shape=[1, inp_image.shape[1], inp_image.shape[2], inp_image.shape[3]]),
                              dtype=tf.float32, trainable=False)
        else:
            mem = 0
        elem, elem2 = tf.image.image_gradients(inp_image)
        elem = tf.cast(elem, tf.float32)
        elem2 = tf.cast(elem2, tf.float32)
        inp_image = tf.cast(inp_image, tf.float32)
        if stream:
            elem = (tf.math.pow(elem, 2)
                    + tf.math.pow(elem2, 2)
                    + tf.math.pow(mem, 2)) / 3
            memor = tf.assign(mem, tf.math.sqrt(elem))
            elem = memor
        else:
            elem = (tf.math.pow(elem, 2)
                    + tf.math.pow(elem2, 2)) / 2
            elem = tf.math.sqrt(elem)
            # elem = elem ** 3
        # elem = tf.clip_by_value(elem, 0, 255)
        # elem = elem / tf.math.reduce_max(elem) * 255
        elem = tf.reduce_max(elem, axis=3)
        elem = elem[:, :, :, tf.newaxis]
        # elem = tf.concat([elem, elem, elem], axis=3)
        # self.elem_u8 = tf.cast(elem, tf.uint8)

        # sobel
        if stream:
            mem_sb = tf.Variable(tf.zeros(shape=[1, inp_image.shape[1], inp_image.shape[2], inp_image.shape[3]]),
                                 dtype=tf.float32, trainable=False)
        else:
            mem_sb = 0
        elem_sb = tf.image.sobel_edges(inp_image)
        elem_sb_x = elem_sb[:, :, :, :, 0]
        elem_sb_y = elem_sb[:, :, :, :, 1]
        if stream:
            elem_sb = (tf.math.pow(elem_sb_x, 2)
                       + tf.math.pow(elem_sb_y, 2)
                       + tf.math.pow(mem_sb, 2)) / 3
            mem_sb = tf.assign(mem_sb, tf.math.sqrt(elem_sb))
            # elem_sb = mem_sb ** 2
            elem_sb = mem_sb
        else:
            elem_sb = (tf.math.pow(elem_sb_x, 2)
                       + tf.math.pow(elem_sb_y, 2)) / 2
            elem_sb = tf.math.sqrt(elem_sb)
            # elem_sb = elem_sb ** 2
        # elem_sb = tf.clip_by_value(elem_sb, 0, 255)
        # elem_sb = elem_sb / tf.math.reduce_max(elem_sb) * 255
        elem_sb = tf.reduce_max(elem_sb, axis=3)
        elem_sb = elem_sb[:, :, :, tf.newaxis]
        # elem_sb = tf.concat([elem_sb, elem_sb, elem_sb], axis=3)
        # self.elem_sb_u8 = tf.cast(elem_sb, tf.uint8)

        # res_elem = (elem + elem_sb) * elem_sb
        res_elem = tf.sqrt(elem ** 2 + elem_sb ** 2) * elem_sb
        res_elem = tf.sqrt(res_elem)
        # res_elem = tf.sqrt(memor ** 2 + mem_sb ** 2) * mem_sb
        # res_elem = elem * elem_sb * res_elem
        # res_elem = res_elem + 50
        res_elem = tf.clip_by_value(res_elem, 100, 150)
        res_elem = res_elem - 100
        res_elem = tf.clip_by_value(res_elem, 0, 255)
        res_elem = res_elem / tf.math.reduce_max(res_elem) * 255
        # self.res_elem_u8 = tf.cast(res_elem, tf.uint8)
        res_res = res_elem

        # hq_res = tf.image.resize_bilinear(res_elem, (960, 1280))
        # self.hq_res_u8 = tf.cast(hq_res, tf.uint8)

        return res_res
