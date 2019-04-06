import numpy as nmp
import cv2 as cv


class CorrMethod:
    """Класс реализующий метод автокорреляции"""
    @staticmethod
    def save_corners(save_path: str, corn):
        """Сохранение координат меток в файл"""
        nmp.save(save_path, corn)

    @staticmethod
    def load_corners(load_path: str):
        """Загрузка координат меток из файла"""
        corn = nmp.load(load_path)

        return corn

    @staticmethod
    def get_corners(src_img,
                    max_corn=100,
                    min_dist=5,
                    qual_lev=0.04):
        """Получить метки"""

        if src_img.shape[-1] == 3:
            image = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            gray = src_img

        # image = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
        # gray = cv.Canny(image, 5, 6)

        # plt.imshow(gray)
        # plt.show()

        corners = cv.goodFeaturesToTrack(gray, max_corn, qual_lev, min_dist)
        corners = corners[:, 0, :]

        return corners

    @staticmethod
    def get_gradient(src_img):
        """Получить градиент"""
        _src_img = src_img.copy()

        out_img = cv.Canny(_src_img, 100, 300)
        out_img = nmp.stack([out_img, out_img, out_img], axis=2)

        return out_img

    @staticmethod
    def get_center(corn, int_type=True):
        """Среднее арифметическое координат всех точек"""
        if int_type:
            avg_coord = nmp.average(corn, axis=0).astype('int32')
        else:
            avg_coord = nmp.average(corn, axis=0)

        return avg_coord

    @staticmethod
    def get_min_length_2(corn_orig, corn_rot):
        """Минимальное расстояние между соседними метками"""
        len_orig = corn_orig.shape[0]
        len_rot = corn_rot.shape[0]

        mat_orig = nmp.reshape(nmp.tile(nmp.reshape(corn_orig, -1), len_rot), [len_orig, len_rot, 2])
        mat_rot = nmp.reshape(nmp.tile(nmp.reshape(corn_rot, -1), len_orig), [len_orig, len_rot, 2])
        mat_rot = nmp.stack([nmp.roll(mat_rot[i], i, axis=0) for i in range(len_orig)], axis=0)

        mat_d = (nmp.sum(nmp.square(mat_rot - mat_orig), axis=2))

        mat_min = nmp.min(mat_d, axis=0)
        quantile = nmp.quantile(mat_min, 0.95)

        mat_sum = nmp.sum(mat_min[mat_min <= quantile])

        return mat_sum

    @staticmethod
    def get_min_length(corn_orig, corn_rot):
        """Минимальное расстояние между соседними метками"""
        mins_orig = nmp.min(corn_orig, axis=0)
        mins_rot = nmp.min(corn_rot, axis=0)
        mins = nmp.abs((nmp.min([mins_orig, mins_rot], axis=0) * 1.1))

        _corn_rot = corn_rot + mins
        _corn_orig = corn_orig + mins

        maxs_orig = nmp.max(_corn_orig, axis=0)
        maxs_rot = nmp.max(_corn_rot, axis=0)
        maxs = (nmp.max([maxs_orig, maxs_rot], axis=0) * 1.1).astype(nmp.int32)

        fill_places = nmp.zeros([maxs[1], maxs[0]])

        cv.fillPoly(fill_places, [_corn_orig.astype(nmp.int32)], 1)
        cv.fillPoly(fill_places, [_corn_rot.astype(nmp.int32)], 0)

        res_fill = nmp.count_nonzero(fill_places)

        # plt.imshow(fill_places[:, :, 1])
        # plt.show()

        return res_fill

    @staticmethod
    def get_dot_plot(corn, loc_image, color):
        """Отображение точек на графике"""
        for dot in corn:
            dot_x, dot_y = dot
            cv.circle(loc_image, (dot_x, dot_y), 3, color, -1)

    @staticmethod
    def start_asc_rot_corn(_corn_orig, _corn_rot, ang_range, ang_step):
        """Перебор перестановок по углу"""
        corn_orig = nmp.copy(_corn_orig)
        corn_rot = nmp.copy(_corn_rot)

        angle = nmp.arange(-ang_range, ang_range, ang_step)
        theta = nmp.radians(angle)
        c, s = nmp.cos(theta), nmp.sin(theta)
        rot_mat = nmp.array(((c, -s), (s, c))).T

        res_mat = nmp.rollaxis(nmp.matmul(rot_mat, corn_rot.T), 2, 1)

        rot_result_length = nmp.zeros([angle.shape[0], 2])
        for i, ang in enumerate(angle):
            rot_result_length[i] = [ang, CorrMethod.get_min_length(corn_orig, res_mat[i])]

        min_len_num = nmp.argmin(rot_result_length, axis=0)[1]
        min_len_res = rot_result_length[min_len_num].astype('int32')

        # min_len_corners = (res_mat[min_len_num] + center_rot).astype('int32')
        min_len_corners = (res_mat[min_len_num]).astype('int32')

        return min_len_res, min_len_corners

    @staticmethod
    def start_asc_pos_corn(_corn_orig, _corn_rot, pos_range, pos_step):
        """Перебор перестановок по осям"""
        corn_orig = nmp.copy(_corn_orig)
        corn_rot = nmp.copy(_corn_rot)

        "Создание матрицы смещений"
        pos_delta = nmp.reshape(nmp.mgrid[-pos_range:pos_range+1:pos_step, -pos_range:pos_range+1:pos_step].T, [-1, 2])

        "Создание матрицы из вектора кординат перемещаемого изображения"
        corn_rot_mat = nmp.rollaxis(nmp.broadcast_to(corn_rot, [pos_delta.shape[0], _corn_rot.shape[0], 2]), 1, 0)

        "Вычитание смещений"
        res_mat = nmp.rollaxis(corn_rot_mat + pos_delta, 1, 0)

        "Расчёт расхождения"
        rot_result_length = nmp.zeros([pos_delta.shape[0], 3])
        for i, x_y in enumerate(pos_delta):
            rot_result_length[i] = [x_y[0], x_y[1], CorrMethod.get_min_length(corn_orig, res_mat[i])]

        "Выбор смещения соответствующего минимальным расхождениям"
        min_len_num = nmp.argmin(rot_result_length, axis=0)[2]
        # min_len_res = (rot_result_length[min_len_num] - [delta_center[0], delta_center[1], 0]).astype('int32')
        min_len_res = (rot_result_length[min_len_num]).astype('int32')

        "Извлечение наиболее удачно подобранного вектора координат"
        # min_len_corners = (res_mat[min_len_num] + center_orig).astype('int32')
        min_len_corners = (res_mat[min_len_num]).astype('int32')

        return min_len_res, min_len_corners

    @staticmethod
    def start_asc_iter_corn(corn_orig, corn_rot, ang_range, ang_step, pos_range, pos_step):
        """Итерация перебора по углам и осям"""
        min_len_res, min_len_corners = CorrMethod.start_asc_rot_corn(corn_orig, corn_rot,
                                                                     ang_range=ang_range, ang_step=ang_step)
        asc_angle = min_len_res

        min_len_res_pos, min_len_corners = CorrMethod.start_asc_pos_corn(corn_orig, min_len_corners,
                                                                         pos_range=pos_range, pos_step=pos_step)
        asc_pos = min_len_res_pos

        return asc_angle, asc_pos, min_len_corners

    @staticmethod
    def asc_constructor(_corn_orig, _corn_rot):
        """Конструктор итераций приближения"""
        corn_orig = nmp.copy(_corn_orig)
        corn_rot = nmp.copy(_corn_rot)

        center_orig = CorrMethod.get_center(corn_orig)
        center_rot = CorrMethod.get_center(corn_rot)
        delta_center = center_rot - center_orig

        corn_orig = corn_orig - center_orig
        corn_rot = corn_rot - center_rot

        # corn_rot = corn_rot - delta_center
        ang = [0, 0]
        pos = -delta_center

        asc_angle, asc_pos, min_len_corners = CorrMethod.start_asc_iter_corn(corn_orig, corn_rot,
                                                                             ang_range=100, ang_step=30,
                                                                             pos_range=2, pos_step=1)
        ang += asc_angle
        pos += asc_pos[:2]

        asc_angle, asc_pos, min_len_corners = CorrMethod.start_asc_iter_corn(corn_orig, min_len_corners,
                                                                             ang_range=30, ang_step=10,
                                                                             pos_range=40, pos_step=4)
        ang += asc_angle
        pos += asc_pos[:2]

        asc_angle, asc_pos, min_len_corners = CorrMethod.start_asc_iter_corn(corn_orig, min_len_corners,
                                                                             ang_range=10, ang_step=1,
                                                                             pos_range=10, pos_step=1)
        ang += asc_angle
        pos += asc_pos[:2]

        min_len_corners = (min_len_corners + center_orig).astype('int32')

        score = asc_pos[2]

        return ang, pos, min_len_corners, score


class ImgContourSegm:
    """Сегментация изображения на основе извлечения контуров"""
    area_treshold = 2000
    error_bound_for_line_percent = 35

    class Line:
        """Представление линии"""
        class Point:
            """Представление точки"""
            def __init__(self, point_x, point_y):
                self.x = point_x
                self.y = point_y

        def __init__(self, line_vect):
            """Ax + By + C = 0"""
            vx, vy, x, y = line_vect

            self.vx_coeff = vx
            self.vy_coeff = vy

            self.pt_1 = self.Point(x, y)
            self.pt_2 = self.Point(x + vx, y + vy)

            self.a_coeff = (self.pt_1.y - self.pt_2.y).item()
            self.b_coeff = (self.pt_2.x - self.pt_1.x).item()
            self.c_coeff = (-self.a_coeff * self.pt_1.x - self.b_coeff * self.pt_1.y).item()

    @staticmethod
    def get_contours(input_img, bgr_flag=False):
        """извлечение контуров"""
        scale_param = int(input_img.shape[1] / 640)
        if scale_param % 2 == 0:
            _input_img = cv.GaussianBlur(input_img, (11 * scale_param + 1, 11 * scale_param + 1), 0.1)
        else:
            _input_img = cv.GaussianBlur(input_img, (11 * scale_param, 11 * scale_param), 0.1)

        if bgr_flag:
            gray = cv.cvtColor(_input_img, cv.COLOR_BGR2GRAY)
        else:
            gray = cv.cvtColor(_input_img, cv.COLOR_RGB2GRAY)
        cnny = cv.Canny(gray, 100, 200)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5 * scale_param, 5 * scale_param))
        cnny2 = cv.morphologyEx(cnny, cv.MORPH_CLOSE, kernel)

        raw_contours = cv.findContours(cnny2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]

        # Отсеивание мелких объёктов и неполных контуров из-за краевых эффектов изображения
        filtered_contours = [local_contour
                             for local_contour in raw_contours
                             if cv.contourArea(local_contour[:, 0]) > ImgContourSegm.area_treshold]
        if len(filtered_contours) == 0:
            outer_contours = ImgContourSegm.damaged_contours_restore(raw_contours)
        else:
            outer_contours = filtered_contours

        # rrr = nmp.zeros_like(input_img)
        # cv.drawContours(rrr, outer_contours, -1, (255, 0, 0))
        # cv.imshow('cnny', cnny)
        # cv.imshow('cnny2', cnny2)
        # cv.imshow('rrr', rrr)
        # plt.imshow(input_img)
        # plt.show()

        return outer_contours

    @staticmethod
    def damaged_contours_restore(list_of_contours):
        """Восстановление контуров, если они оказались разорваны"""
        # Определение параметров получения производной
        roll_len = 4
        # Определение свойств окна для свёртки
        window_length = 9
        w_hl = window_length // 2
        conv_window = nmp.ones(window_length)

        # Отсечение неподходящих контуров
        filtered_contours = [local_contour
                             for local_contour in list_of_contours
                             if cv.contourArea(local_contour) < cv.arcLength(local_contour, False)]

        result_of_restore = []

        for contour in filtered_contours:
            # Получение первой производной по обеим осям и сглаживание
            array_of_deltas_f1 = ((contour[:, 0] - nmp.roll(contour[:, 0], roll_len, axis=0)) / roll_len)
            array_of_deltas_f1 = nmp.roll(array_of_deltas_f1, -roll_len, axis=0)
            array_of_deltas_f1 = nmp.sqrt(nmp.sum(nmp.square(array_of_deltas_f1), axis=1))

            smooth_arr_of_deltas_f1 = nmp.pad(array_of_deltas_f1, (w_hl, w_hl), 'wrap')
            smooth_arr_of_deltas_f1 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f1,
                                                           conv_window, mode='valid') / window_length)
            smooth_arr_of_deltas_f1 = nmp.pad(smooth_arr_of_deltas_f1, (w_hl, w_hl), 'wrap')
            smooth_arr_of_deltas_f1 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f1,
                                                           conv_window, mode='valid') / window_length)

            # Определение перелома в получившейся зависимости
            cliped_array = (smooth_arr_of_deltas_f1 < 0.9).astype(nmp.int)
            cliped_array = nmp.clip(nmp.roll(cliped_array, 1, axis=0) - cliped_array, 0, 1)
            ind_to_delete = nmp.nonzero(cliped_array)
            if ind_to_delete[0].size > 0:
                if len(ind_to_delete) != 2:
                    # return []
                    continue
                new_contour = nmp.delete(contour, nmp.arange(ind_to_delete[0][0], ind_to_delete[0][1]), axis=0)
            else:
                new_contour = contour

            if cv.contourArea(new_contour[:, 0]) > ImgContourSegm.area_treshold:
                result_of_restore.append(new_contour)

            # plt.subplot(2, 2, 1)
            # plt.plot(new_contour[:, 0])
            # plt.subplot(2, 2, 2)
            # plt.plot(array_of_deltas_f1)
            # plt.subplot(2, 2, 3)
            # plt.plot(cliped_array)
            # plt.subplot(2, 2, 4)
            # img = nmp.zeros([480, 640])
            # cv.drawContours(img, [new_contour], -1, 1)
            # plt.imshow(img)
            # plt.show()

        output_stack = result_of_restore

        return output_stack

    @staticmethod
    def get_detecting_plot_from_contour(axis_points):
        """Извлечение вектора второй производной от проекции точек на ось"""
        # Определение параметров получения производной
        # roll_len = 4
        roll_len = int(axis_points.shape[0] * 0.05)
        # Определение свойств окна для свёртки
        # window_length = 9
        window_length = int(axis_points.shape[0] * 0.02)
        w_hl = window_length // 2
        conv_window = nmp.ones(window_length)

        # Получение первой производной и сглаживание
        array_of_deltas_f1 = ((axis_points - nmp.roll(axis_points, roll_len, axis=0)) / roll_len)
        array_of_deltas_f1 = nmp.roll(array_of_deltas_f1, -roll_len//2, axis=0)

        smooth_arr_of_deltas_f1 = nmp.pad(array_of_deltas_f1, (w_hl, w_hl), 'wrap')
        smooth_arr_of_deltas_f1 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f1,
                                                       conv_window, mode='valid') / window_length)
        smooth_arr_of_deltas_f1 = nmp.pad(smooth_arr_of_deltas_f1, (w_hl, w_hl), 'wrap')
        smooth_arr_of_deltas_f1 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f1,
                                                       conv_window, mode='valid') / window_length)

        # Получение второй производной и сглаживание
        array_of_deltas_f2 = (smooth_arr_of_deltas_f1 - nmp.roll(smooth_arr_of_deltas_f1, roll_len, axis=0)) / roll_len
        array_of_deltas_f2 = nmp.roll(array_of_deltas_f2, -roll_len//2, axis=0)

        smooth_arr_of_deltas_f2 = nmp.pad(array_of_deltas_f2, (w_hl, w_hl), 'wrap')
        smooth_arr_of_deltas_f2 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f2,
                                                       conv_window, mode='valid') / window_length)
        smooth_arr_of_deltas_f2 = nmp.pad(smooth_arr_of_deltas_f2, (w_hl, w_hl), 'wrap')
        smooth_arr_of_deltas_f2 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f2,
                                                       conv_window, mode='valid') / window_length)

        # Определение порогового значения и применение к полученному вектору
        # treshold_val = (nmp.max(smooth_arr_of_deltas_f2) + nmp.min(smooth_arr_of_deltas_f2)) / 2
        # treshold_val = 0.01
        treshold_val = nmp.quantile(smooth_arr_of_deltas_f2, 0.85)
        output_bool_detect = smooth_arr_of_deltas_f2 > treshold_val

        # plt.subplot(2, 2, 1)
        # plt.plot(smooth_arr_of_deltas_f1)
        # plt.subplot(2, 2, 2)
        # plt.plot(smooth_arr_of_deltas_f2)
        # plt.plot([nmp.median(smooth_arr_of_deltas_f2)]*len(smooth_arr_of_deltas_f2), color='red')
        # plt.plot([nmp.mean(smooth_arr_of_deltas_f2)] * len(smooth_arr_of_deltas_f2), color='green')
        # plt.plot([treshold_val]*len(smooth_arr_of_deltas_f2), color='brown')
        # plt.subplot(2, 2, 3)
        # plt.hist(smooth_arr_of_deltas_f2, 100, [0, nmp.max(smooth_arr_of_deltas_f2)])
        # plt.subplot(2, 2, 4)
        # plt.plot(output_bool_detect)
        # plt.show()

        return output_bool_detect

    @staticmethod
    def get_detecting_plot_from_contour_2(axis_points):
        """Извлечение вектора второй производной"""
        # Определение параметров получения производной
        roll_len = int(axis_points.shape[0] * 0.05)
        # Определение свойств окна для свёртки
        window_length = int(axis_points.shape[0] * 0.02)
        w_hl = window_length // 2
        conv_window = nmp.ones(window_length)

        # Получение первой производной и сглаживание
        array_of_deltas_f1 = ((axis_points - nmp.roll(axis_points, roll_len, axis=0)) / roll_len)
        array_of_deltas_f1 = nmp.roll(array_of_deltas_f1, -roll_len//2, axis=0)
        array_of_deltas_f1 = nmp.sqrt(nmp.sum(nmp.square(array_of_deltas_f1), axis=1))

        smooth_arr_of_deltas_f1 = nmp.pad(array_of_deltas_f1, (w_hl, w_hl), 'wrap')
        smooth_arr_of_deltas_f1 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f1,
                                                       conv_window, mode='valid') / window_length)
        smooth_arr_of_deltas_f1 = nmp.pad(smooth_arr_of_deltas_f1, (w_hl, w_hl), 'wrap')
        smooth_arr_of_deltas_f1 = nmp.abs(nmp.convolve(smooth_arr_of_deltas_f1,
                                                       conv_window, mode='valid') / window_length)

        # Определение порогового значения и применение к полученному вектору
        treshold_val = nmp.quantile(smooth_arr_of_deltas_f1, 0.20)
        output_bool_detect = smooth_arr_of_deltas_f1 < treshold_val

        return output_bool_detect

    @staticmethod
    def get_seq_from_line_plot(inp_vec):
        """Получение последовательностей индексов представляющих прямые линиин на графике контуров"""
        # Приведение типов к числовому значению
        integer_inp_vec = inp_vec.astype(nmp.int)

        # Проверка на наличие хотя бы одной прямой
        if nmp.sum(integer_inp_vec) == 0:
            return None

        # Получение первой производной для обнаружения фронтов
        array_of_deltas_f1 = integer_inp_vec - nmp.roll(integer_inp_vec, 1, axis=0)
        array_of_deltas_f1 = nmp.roll(array_of_deltas_f1, -1, axis=0)

        # Разделение фронтов в два разных массива
        rising_fronts = nmp.clip(array_of_deltas_f1, 0, 1)
        falling_fronts = nmp.clip(array_of_deltas_f1, -1, 0)

        # Получение индексов соответствующих фронтам на имеющихся проекциях
        rising_fronts_ids = nmp.nonzero(rising_fronts)[0]
        falling_fronts_ids = nmp.nonzero(falling_fronts)[0]

        # Если начало отрезка попало в конец массива точек, то происходит сдвиг и переброс через границу
        if falling_fronts_ids[0] > rising_fronts_ids[0]:
            falling_fronts_ids = nmp.roll(falling_fronts_ids, 1, axis=0)
            falling_fronts_ids[0] = falling_fronts_ids[0] - falling_fronts.shape[0]

        # Объединение индексов начала и конца отрезков в один пакет
        lines_list = nmp.stack([falling_fronts_ids, rising_fronts_ids]).T

        return lines_list

    @staticmethod
    def get_lines_from_img(input_image, bgr_flag=False):
        """Полный алгоритм извлечения прямых линий из изображения"""
        # Получение контуров
        contours = ImgContourSegm.get_contours(input_image, bgr_flag)
        num_of_contours = len(contours)

        # Получение прямых для каждого контура в отдельности
        detect_lines = []
        for one_contour in contours:
            # Детектирование прямых по каждой оси в отдельности и объединение результатов
            axis_detect = ImgContourSegm.get_detecting_plot_from_contour_2(one_contour[:, 0])
            # axis_view_1 = ImgContourSegm.get_detecting_plot_from_contour(one_contour[:, 0, 0])
            # axis_view_2 = ImgContourSegm.get_detecting_plot_from_contour(one_contour[:, 0, 1])
            # axis_detect = axis_view_1 | axis_view_2

            # plt.plot(axis_detect)
            # plt.show()

            # Преобразование массива-маски к набору координат в массиве точек контура
            lines_from_this_contour = ImgContourSegm.get_seq_from_line_plot(axis_detect)

            # Если прямых не было обнаружено - пропуск текущего контура
            if lines_from_this_contour is None:
                continue

            # Получение срезов из массива точек данного контура по заданным границам
            line_points_array = []
            for line_pos in lines_from_this_contour:
                delta_pos = nmp.abs(line_pos[1] - line_pos[0]) / 100

                center = (line_pos[0] + line_pos[1]) / 2

                start_edge = int(center - delta_pos * ImgContourSegm.error_bound_for_line_percent)
                end_edge = int(center + delta_pos * ImgContourSegm.error_bound_for_line_percent)

                if start_edge < 0:
                    rolled_array = nmp.roll(one_contour[:, 0, :], -start_edge, axis=0)
                    line_points_array.append(rolled_array[0:end_edge - start_edge])
                else:
                    line_points_array.append(one_contour[start_edge:end_edge, 0, :])

            detect_lines.append(nmp.array(line_points_array))

        return num_of_contours, detect_lines, contours

    @staticmethod
    def draw_lines(input_image, list_of_lines_by_contours, thick=2):
        """Отрисовка всех полученных линий"""
        for lines_of_contour in list_of_lines_by_contours:
            rand_color = (0, 255, 0)
            for line in lines_of_contour:
                if line.size == 0:
                    continue
                cv.line(input_image, tuple(line[0]), tuple(line[-1]), rand_color, thick)

    @staticmethod
    def compute_lines_with_rotation(angles_by_contours, list_of_lines_by_contours):
        """Пересчёт точек каждой линии для каждого контура в соответствии с полученным углом и расчёт центра прямой"""
        loc_contour_stack_line = []
        loc_contour_stack_mean = []
        # Перебор наборов прямых по контурам
        for num, lines_of_contour in enumerate(list_of_lines_by_contours):
            # Получение матрицы поворота
            theta = nmp.radians(angles_by_contours[num])
            c, s = nmp.cos(theta), nmp.sin(theta)
            rotation_mat = nmp.array(((c, -s), (s, c))).T

            loc_line_stack = []
            loc_mean_stack = []
            # Перебор всех прямых в наборе для этого контура
            for line in lines_of_contour:
                # Если прямой нет, то пропуск
                if line.size == 0:
                    continue
                # Применение поворота, вычисление центра прямой, получение длин проекций данного отрезка на оси
                rot_result = nmp.rollaxis(nmp.matmul(rotation_mat, line.T), 1, 0)
                mean_result = nmp.mean(rot_result, axis=0)
                vec_result = rot_result[-1] - rot_result[0]

                loc_line_stack.append(vec_result)
                loc_mean_stack.append(mean_result)
            loc_contour_stack_line.append(loc_line_stack)
            loc_contour_stack_mean.append(loc_mean_stack)

        return loc_contour_stack_line, loc_contour_stack_mean

    @staticmethod
    def choose_lines_for_intersect(mean_lines_by_contour, lines_by_contours):
        """Выбор из всего набора прямых только тех, что нужны для определения точки пересечения и угла наклона"""
        found_lines = []
        found_for_angle = []
        # Перебор наборов прямых по контурам
        for num, contour_lines in enumerate(mean_lines_by_contour):
            # Инверсия компоненты высоты, т.к. меньшие значения наверху
            loc_lines = nmp.array(contour_lines)
            loc_lines[:, 1] = nmp.max(loc_lines[:, 1]) - loc_lines[:, 1]
            # Минимум по центрам всех прямых
            min_cr = nmp.min(loc_lines, axis=0)

            # Получение радиус-вектора от точки минимума до центра каждой прямой
            lengths_of_vect = nmp.sqrt(nmp.sum(nmp.square(loc_lines - min_cr), axis=1))
            # Сортировка и добавление ближайших линий в список
            sorted_l = nmp.argsort(lengths_of_vect)
            choosen_lines = [lines_by_contours[num][sorted_l[0]], lines_by_contours[num][sorted_l[1]]]
            found_lines.append(choosen_lines)

            # Выбор из полученных прямых самой нижней
            for_angle_array = loc_lines[[sorted_l[0], sorted_l[1]], 1]
            for_angle = choosen_lines[nmp.argmin(for_angle_array).item()]
            found_for_angle.append(for_angle)

        return found_lines, found_for_angle

    @staticmethod
    def get_intersect_point(lines_by_contours):
        """Расчёт точки пересечения прямых и определение нуля детали"""
        intersect_point = []
        for lines in lines_by_contours:
            """Ax + By + C = 0"""
            if lines[0].size == 0:
                intersect_point.append(lines[1][-1])
                continue
            elif lines[1].size == 0:
                intersect_point.append(lines[0][-1])
                continue

            # Расчёт точки пересечения полученных прямых
            line_1 = ImgContourSegm.Line(cv.fitLine(lines[0], cv.DIST_L2, 0, 0.01, 0.01))
            line_2 = ImgContourSegm.Line(cv.fitLine(lines[1], cv.DIST_L2, 0, 0.01, 0.01))

            mat_low_det = nmp.array([[line_1.a_coeff, line_1.b_coeff],
                                     [line_2.a_coeff, line_2.b_coeff]])
            low_det = nmp.linalg.det(mat_low_det)

            if low_det == 0:
                continue

            mat_x_up_det = nmp.array([[line_1.c_coeff, line_1.b_coeff],
                                     [line_2.c_coeff, line_2.b_coeff]])
            x_up_det = nmp.linalg.det(mat_x_up_det)

            mat_y_up_det = nmp.array([[line_1.a_coeff, line_1.c_coeff],
                                      [line_2.a_coeff, line_2.c_coeff]])
            y_up_det = nmp.linalg.det(mat_y_up_det)

            x_point = int(- x_up_det / low_det)
            y_point = int(- y_up_det / low_det)

            intersect_point.append([x_point, y_point])

        return intersect_point

    @staticmethod
    def get_angle_2(precompute_angles, lines_by_contours):
        """Получение угла наклона по выбранной прямой"""
        angles = []
        # Перебор прямых по контурам
        for num, contour_line in enumerate(lines_by_contours):
            # Проверка, найдена ли нужная прямая
            if contour_line.size == 0:
                angles.append(precompute_angles[num])
                continue
            # Аппроксимация точек прямой
            vx, vy, x, y = cv.fitLine(contour_line, cv.DIST_L2, 0, 0.01, 0.01)

            # Проверка, параллельна ли прямая одной из осей
            correct_check = [vx.item(), vy.item()]
            if correct_check[0] == 0:
                angles.append(90)
                continue
            elif correct_check[1] == 0:
                angles.append(0)
            else:
                # Вычисление угла наклона прямой
                atan_2 = nmp.arctan2(correct_check[1], correct_check[0])
                angle = nmp.rad2deg(atan_2)
                angles.append(angle)

        return angles

    @staticmethod
    def get_angle(precompute_angles, lines_by_contours):
        """Получение угла наклона по выбранной прямой"""
        angles = []
        # Перебор прямых по контурам
        for num, contour_line in enumerate(lines_by_contours):
            # Проверка, найдена ли нужная прямая
            if contour_line.size == 0:
                angles.append(precompute_angles[num])
                continue
            # Точки интереса - начало и конец прямой
            pt_1 = contour_line[0]
            pt_2 = contour_line[-1]

            # Проверка, параллельна ли прямая одной из осей
            correct_check = pt_2 - pt_1
            if correct_check[0] == 0:
                angles.append(90)
                continue
            elif correct_check[1] == 0:
                angles.append(0)
            else:
                # Вычисление угла наклона прямой
                atan_2 = nmp.arctan2(correct_check[1], correct_check[0])
                angle = nmp.rad2deg(atan_2)
                angles.append(angle)
                # gipp = nmp.sqrt(nmp.sum(nmp.square(correct_check)))
                # sin_f = correct_check[1] / gipp
                # arc_sin = nmp.arcsin(sin_f)
                # angles.append(nmp.rad2deg(arc_sin))

        return angles


class MainPositionClass:
    """Класс реализующий все необходимые для позиционирования алгоритмы"""
    image_size_for_storage = (640, 480)

    def __init__(self, original_images_for_classes: list, metric_params_list: list):
        """Инициализация класса с получением представителей каждого из классов, включая фон"""
        # Сохранение уменьшенных копий оригинальных изображений
        self.img_orig_storage = []
        self.img_orig_canny_storage = []
        for image in original_images_for_classes:
            resized_image = cv.resize(image, self.image_size_for_storage)
            canny_image = cv.Canny(cv.cvtColor(resized_image, cv.COLOR_RGB2GRAY), 100, 200)
            canny_image = cv.morphologyEx(canny_image, cv.MORPH_CLOSE, nmp.ones([3, 3]))
            self.img_orig_storage.append(resized_image)
            self.img_orig_canny_storage.append(canny_image)

        # Установка параметров преобразования пиксельных единиц в метрические
        self.image_metric_params = {}
        self.measure_linears = nmp.zeros_like(self.img_orig_storage[0])
        self.mm_by_pixel = 1
        self.scale_ratio = 1
        det_pixel_len, detail_mm_len, orig_img_size, scale_mm_len = metric_params_list
        self.recompile_metric_params(det_pixel_len, detail_mm_len, orig_img_size, scale_mm_len)

        # Сохранение контуров оригинальных изображений
        self.orig_corner_storage = []
        self.orig_area_storage = []
        for num, image in enumerate(self.img_orig_storage):
            if num == 0:
                self.orig_corner_storage.append(None)
                continue
            orig_corn = self.get_corners_operations(image)[0]
            self.orig_corner_storage.append(orig_corn)
            self.orig_area_storage.append(cv.contourArea(orig_corn))

            # rrr = nmp.zeros_like(image)
            # cv.drawContours(rrr, orig_corn, -1, (255, 0, 0))
            # plt.imshow(rrr)
            # plt.show()

        # Установка нижней границы площади детали
        minimal_area = min(self.orig_area_storage) * 0.7
        ImgContourSegm.area_treshold = int(minimal_area)

        # Списки для хранения текущего набора котуров и уменьшенной копии
        self.target_contours_storage = []
        self.target_minimize_storage = nmp.zeros_like(self.img_orig_storage[0])
        self.target_fully_storage = nmp.zeros_like(self.img_orig_storage[0])
        self.target_min_contours_storage = []

    def first_step(self, frame):
        """Первый шаг обработки: получение кадра и выдача центрированной копии каждой детали для классификации
            Помимо этого, сохранение набора контуров всех деталей на изображении в хранилище"""
        img_for_classify_list = []

        # Уменьшение размера
        minimized_frame = self.resize_input_img(frame)
        frame_contours = self.get_corners_operations(frame)
        minimized_frame_contours = self.get_corners_operations(minimized_frame)

        # Проверка на наличие деталей
        if len(frame_contours) == 0:
            return None

        # Извлечение нормированных изображений
        for single_contour in frame_contours:
            img_for_classify = self.get_image_for_classification(single_contour, minimized_frame)
            img_for_classify_list.append(img_for_classify)

        # Сохранение контуров и миниатюр
        self.target_contours_storage = frame_contours
        self.target_minimize_storage = minimized_frame
        self.target_fully_storage = frame
        self.target_min_contours_storage = minimized_frame_contours

        return img_for_classify_list

    def second_step_2(self, classify_result_list: list):
        """Второй шаг обработки: подгрузка нужной заготовки и определение координат детали"""
        angle_extracted = []

        # Корреляционный подбор
        for num, contour in enumerate(self.target_contours_storage):
            target_class_num = classify_result_list[num]
            orig_contour = self.orig_corner_storage[target_class_num]
            orig_canny = self.img_orig_canny_storage[target_class_num]

            ang, pos, min_len_corners, score = CorrMethod.asc_constructor(orig_contour, contour[::10, 0, :])

            # К полярным координатам
            small_target_image = self.target_minimize_storage
            rot_mat = cv.getRotationMatrix2D(tuple(CorrMethod.get_center(contour[:, 0, :])), ang[0], 1)
            warped_target_image = cv.warpAffine(small_target_image, rot_mat, self.image_size_for_storage)
            rolled_target_image = nmp.roll(warped_target_image, [pos[0], pos[1]], axis=[1, 0])

            target_canny = cv.Canny(cv.cvtColor(rolled_target_image, cv.COLOR_RGB2GRAY), 100, 200)

            center_x, center_y = nmp.mean(min_len_corners, axis=0).astype(nmp.int)
            pol_img_orig = cv.linearPolar(orig_canny, (center_x, center_y), 150,
                                          cv.WARP_FILL_OUTLIERS).astype(nmp.float32)
            pol_img_target = cv.linearPolar(target_canny, (center_x, center_y), 150,
                                            cv.WARP_FILL_OUTLIERS).astype(nmp.float32)[0:240]

            match_plot = cv.matchTemplate(pol_img_orig, pol_img_target, cv.TM_CCORR_NORMED)
            plot_x_axis = nmp.linspace(0, 180, len(match_plot))

            result = nmp.argmax(match_plot)
            add_angle = plot_x_axis[result]

            loc_angle = -(ang[0] + add_angle)
            if loc_angle < 0:
                loc_angle = loc_angle + 360.

            angle_extracted.append(-loc_angle)

        # Получение прямых, установка нуля и определение угла
        num_of_details, res_lines, res_contours = ImgContourSegm.get_lines_from_img(self.target_fully_storage)

        filtered_lines, lines_center = ImgContourSegm.compute_lines_with_rotation(angle_extracted, res_lines)
        result_lines_by_contours, result_angle_lines_by_contours = \
            ImgContourSegm.choose_lines_for_intersect(lines_center, res_lines)

        points = ImgContourSegm.get_intersect_point(result_lines_by_contours)
        angle = ImgContourSegm.get_angle(angle_extracted, result_angle_lines_by_contours)

        coord_list = [points, angle]
        finding_lines = res_lines

        return coord_list, finding_lines

    def second_step(self, classify_result_list: list):
        """Второй шаг обработки: подгрузка нужной заготовки и определение координат детали"""
        angle_extracted = []

        # Поиск по шаблону
        for num, contour in enumerate(self.target_contours_storage):
            target_class_num = classify_result_list[num]
            orig_image = self.img_orig_storage[target_class_num]
            orig_contour = self.orig_corner_storage[target_class_num]
            # orig_area = self.orig_area_storage[target_class_num]

            if target_class_num == 0:
                angle_extracted.append(0)
                continue

            # # К полярным координатам
            small_target_image = self.target_minimize_storage
            small_target_contour = self.target_min_contours_storage[num]

            loc_angle = self.to_polar_and_find_template_2(small_target_contour, orig_contour,
                                                          orig_image, small_target_image)

            angle_extracted.append(loc_angle)

        # Получение прямых, установка нуля и определение угла
        coord_list, res_lines = self.positioning(self.target_fully_storage, angle_extracted)

        finding_lines = res_lines

        return coord_list, finding_lines

    def third_step(self, coord_list, finding_lines, detail_names: list):
        """Третий шаг: отрисовка частей"""
        result_image = self.target_fully_storage.copy()
        result_image = cv.cvtColor(result_image, cv.COLOR_RGB2BGR)
        result_image[self.measure_linears] = (0, 0, 255)
        for num, pt in enumerate(coord_list[0]):
            angle = coord_list[1][num]
            result_image = self.drawing_operations(result_image, pt, angle,
                                                   [finding_lines[num]], num, str(detail_names[num]))

        result_image = cv.cvtColor(result_image, cv.COLOR_RGB2BGR)
        return result_image

    def recompile_metric_params(self, det_pixel_len, detail_mm_len, orig_img_size, scale_mm_len):
        """Переобпределение параметров изображения в метрических единицах"""
        # Словарь с необходимыми для преобразования величин значениями
        # default: [220, 75.54, (480, 640), 10]
        self.image_metric_params = {
            'detail_pixel_length':      det_pixel_len,
            'detail_mm_length':         detail_mm_len,
            'orig_image_size':          orig_img_size,
            'orig_image_half_size':     (orig_img_size[0] // 2, orig_img_size[1] // 2),
            'scale_mm_length':          scale_mm_len
        }

        # Разрешение изображения в мм на пиксель и цена деления сетки
        self.mm_by_pixel = self.image_metric_params['detail_mm_length'] / self.image_metric_params[
            'detail_pixel_length']
        self.scale_ratio = int(self.image_metric_params['scale_mm_length'] / self.mm_by_pixel)

        # Создание координатной сетки
        image_h, image_w = self.image_metric_params['orig_image_size']
        im_half_h, im_half_w = self.image_metric_params['orig_image_half_size']

        linears = nmp.zeros([image_h, image_w])
        for y_line in range(im_half_h, image_h, self.scale_ratio):
            cv.line(linears, (0, y_line), (image_w, y_line), 1)
        for y_line in range(im_half_h, 0, -self.scale_ratio):
            cv.line(linears, (0, y_line), (image_w, y_line), 1)

        for x_line in range(im_half_w, image_w, self.scale_ratio):
            cv.line(linears, (x_line, 0), (x_line, image_h), 1)
        for x_line in range(im_half_w, 0, -self.scale_ratio):
            cv.line(linears, (x_line, 0), (x_line, image_h), 1)

        cv.line(linears, (0, im_half_h), (image_w, im_half_h), 1, 2)
        cv.line(linears, (im_half_w, 0), (im_half_w, image_h), 1, 2)
        self.measure_linears = linears.astype(nmp.bool)

    def resize_input_img(self, _input_image):
        """Сжатие изображения"""
        input_image = _input_image.copy()
        small_h = self.image_size_for_storage[1]
        # small_w = int(nmp.round(input_image.shape[1] / (input_image.shape[0] / small_h)))
        small_w = self.image_size_for_storage[0]
        small_target_image = cv.resize(input_image, (small_w, small_h), interpolation=cv.INTER_NEAREST)

        return small_target_image

    @staticmethod
    def get_corners_operations(input_image):
        """Получение точек контура"""
        output_corners = ImgContourSegm.get_contours(input_image)

        return output_corners

    @staticmethod
    def compute_correlate(orig_contour, target_contour):
        """Корреляционный подбор"""
        ang, pos, min_len_corners, score = CorrMethod.asc_constructor(orig_contour[::10, 0, :],
                                                                      target_contour[::10, 0, :])

        return ang, pos, min_len_corners, score

    @staticmethod
    def to_polar_and_find_template(contour_for_rot,
                                   given_angle, given_pos, min_len_corners,
                                   orig_image, small_target_image):
        """К полярным координатам"""
        # Определение свойств окна для свёртки
        window_length = 21
        w_hl = window_length // 2
        conv_window = nmp.ones(window_length)

        rot_mat = cv.getRotationMatrix2D(tuple(CorrMethod.get_center(contour_for_rot[:, 0, :])), given_angle[0], 1)
        dst = cv.warpAffine(small_target_image, rot_mat, (640, 480))

        dst2 = nmp.roll(dst, [given_pos[0], given_pos[1]], axis=[1, 0])

        # _image_1_ = cv.Canny(cv.cvtColor(orig_image, cv.COLOR_RGB2GRAY), 100, 200)
        # _image_2_ = cv.Canny(cv.cvtColor(dst2, cv.COLOR_RGB2GRAY), 100, 200)

        _image_1_ = cv.cvtColor(orig_image, cv.COLOR_RGB2GRAY)
        _image_2_ = cv.cvtColor(dst2, cv.COLOR_RGB2GRAY)

        _image_1_ = cv.GaussianBlur(_image_1_, (9, 9), 0.5)
        _image_2_ = cv.GaussianBlur(_image_2_, (9, 9), 0.5)

        center_x, center_y = nmp.mean(min_len_corners, axis=0).astype(nmp.int)
        pol_img_1 = cv.linearPolar(_image_1_,
                                   (center_x, center_y), 150, cv.WARP_FILL_OUTLIERS).astype(nmp.float32)
        pol_img_1 = nmp.concatenate([pol_img_1, pol_img_1], axis=0)
        pol_img_2 = cv.linearPolar(_image_2_,
                                   (center_x, center_y), 150, cv.WARP_FILL_OUTLIERS).astype(nmp.float32)[:, 140:500]

        # plt.subplot(1, 2, 1)
        # plt.imshow(pol_img_1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(pol_img_2)
        # plt.show()

        ret = cv.matchTemplate(pol_img_1, pol_img_2, cv.TM_CCORR_NORMED)
        ret = nmp.max(ret, axis=1)
        # ret = ret[:, 0]

        # x = nmp.linspace(0, 360, len(ret))
        # plt.plot(x, ret)
        # plt.show()

        smooth_ret = nmp.pad(ret, (w_hl, w_hl), 'wrap')
        smooth_ret = nmp.abs(nmp.convolve(smooth_ret,
                                          conv_window, mode='valid') / window_length)

        smooth_ret = nmp.pad(smooth_ret, (w_hl, w_hl), 'wrap')
        smooth_ret = nmp.abs(nmp.convolve(smooth_ret,
                                          conv_window, mode='valid') / window_length)

        smooth_ret = nmp.pad(smooth_ret, (w_hl, w_hl), 'wrap')
        smooth_ret = nmp.abs(nmp.convolve(smooth_ret,
                                          conv_window, mode='valid') / window_length)

        # x = nmp.linspace(0, 360, len(ret))
        x_2 = nmp.linspace(0, 360, len(smooth_ret))

        # plt.plot(x, ret)
        # plt.plot([nmp.max(smooth_ret)] * 360)
        # plt.plot(x_2, smooth_ret)
        # plt.show()

        # result = nmp.argmax(ret)
        # add_angle = x[result]

        result = nmp.argmax(smooth_ret)
        add_angle = x_2[result]

        loc_angle = -(given_angle[0] + add_angle)
        if loc_angle < 0:
            loc_angle = loc_angle + 360.

        return -loc_angle

    @staticmethod
    def to_polar_and_find_template_2(contour_for_rot, contour_orig,
                                     orig_image, small_target_image):
        """К полярным координатам"""
        # Определение свойств окна для свёртки
        window_length = 21
        w_hl = window_length // 2
        conv_window = nmp.ones(window_length)

        _image_1_ = cv.Canny(cv.cvtColor(orig_image, cv.COLOR_RGB2GRAY), 100, 200)
        _image_2_ = cv.Canny(cv.cvtColor(small_target_image, cv.COLOR_RGB2GRAY), 100, 200)

        # plt.subplot(1, 2, 1)
        # plt.imshow(_image_1_)
        # plt.subplot(1, 2, 2)
        # plt.imshow(_image_2_)
        # plt.show()

        # _image_1_ = cv.cvtColor(orig_image, cv.COLOR_RGB2GRAY)
        # _image_2_ = cv.cvtColor(small_target_image, cv.COLOR_RGB2GRAY)

        _image_1_ = cv.GaussianBlur(_image_1_, (9, 9), 0.5)
        _image_2_ = cv.GaussianBlur(_image_2_, (9, 9), 0.5)

        (orig_center_x, orig_center_y), orig_r = cv.minEnclosingCircle(contour_orig)
        (target_center_x, target_center_y), target_r = cv.minEnclosingCircle(contour_for_rot)

        pol_img_1 = cv.linearPolar(_image_1_,
                                   (int(orig_center_x), int(orig_center_y)), orig_r,
                                   cv.WARP_FILL_OUTLIERS).astype(nmp.float32)
        pol_img_1 = nmp.concatenate([pol_img_1, pol_img_1], axis=0)

        pol_img_2 = cv.linearPolar(_image_2_,
                                   (int(target_center_x), int(target_center_y)), target_r,
                                   cv.WARP_FILL_OUTLIERS).astype(nmp.float32)[:, 140:500]

        # plt.subplot(1, 2, 1)
        # plt.imshow(pol_img_1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(pol_img_2)
        # plt.show()

        ret = cv.matchTemplate(pol_img_1, pol_img_2, cv.TM_CCORR_NORMED)
        ret = nmp.max(ret, axis=1)
        # ret = ret[:, 0]

        # x = nmp.linspace(0, 360, len(ret))
        # plt.plot(x, ret)
        # plt.show()

        # iii = cv.drawContours(small_target_image, contour_for_rot, -1, (255, 0, 0))
        # plt.imshow(iii)
        # plt.show()

        smooth_ret = nmp.pad(ret, (w_hl, w_hl), 'wrap')
        smooth_ret = nmp.abs(nmp.convolve(smooth_ret,
                                          conv_window, mode='valid') / window_length)

        smooth_ret = nmp.pad(smooth_ret, (w_hl, w_hl), 'wrap')
        smooth_ret = nmp.abs(nmp.convolve(smooth_ret,
                                          conv_window, mode='valid') / window_length)

        smooth_ret = nmp.pad(smooth_ret, (w_hl, w_hl), 'wrap')
        smooth_ret = nmp.abs(nmp.convolve(smooth_ret,
                                          conv_window, mode='valid') / window_length)

        # x = nmp.linspace(0, 360, len(ret))
        x_2 = nmp.linspace(0, 360, len(smooth_ret))

        # plt.plot(x, ret)
        # plt.plot([nmp.max(smooth_ret)] * 360)
        # plt.plot(x_2, smooth_ret)
        # plt.show()

        # result = nmp.argmax(ret)
        # add_angle = x[result]

        result = nmp.argmax(smooth_ret)
        add_angle = x_2[result]

        # loc_angle = -add_angle
        # if loc_angle < 0:
        #     loc_angle = loc_angle + 360.

        return -add_angle

    @staticmethod
    def positioning(full_target_image, given_angle):
        """Получение прямых и установка нуля детали"""
        num_of_details, res_lines, res_contours = ImgContourSegm.get_lines_from_img(full_target_image)

        res_li, res_me = ImgContourSegm.compute_lines_with_rotation(given_angle, res_lines)

        result_lines_by_contours, result_angle_lines_by_contours = ImgContourSegm.choose_lines_for_intersect(res_me,
                                                                                                             res_lines)

        points = ImgContourSegm.get_intersect_point(result_lines_by_contours)
        angles = ImgContourSegm.get_angle_2(given_angle, result_angle_lines_by_contours)

        return (points, angles), res_lines

    def drawing_operations(self, target_image, points, angle, lines, number_of_contour, name_of_detail):
        """Отрисовка текста и сетки"""
        new_image = target_image.copy()

        scale_factor = new_image.shape[0] // 480
        cv.circle(new_image, tuple(points), 2 * scale_factor, (50, 50, 255), -1)

        text = '{0} - Y: {1:.2f}mm, X: {2:.2f}mm, ANG: {3:.2f}' \
            .format(name_of_detail,
                    (-points[1] + new_image.shape[0] // 2) * self.mm_by_pixel,
                    (points[0] - new_image.shape[1] // 2) * self.mm_by_pixel,
                    -angle)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(new_image, text,
                   (10 * scale_factor, 20 * scale_factor * (number_of_contour + 1)),
                   font, 0.5 * scale_factor, (255, 0, 0), 1 * scale_factor, cv.LINE_AA)

        ImgContourSegm.draw_lines(new_image, lines, 2 * scale_factor)

        self.draw_axis(new_image, points, angle)

        return new_image

    def draw_axis(self, image, pos, angle):
        """Отрисовка осей"""
        axis_length_mm = 25
        axis_length_pix = axis_length_mm / self.mm_by_pixel

        angle_rad = nmp.deg2rad(angle)
        axis_x_view = nmp.cos(angle_rad) * axis_length_pix
        axis_y_view = nmp.sin(angle_rad) * axis_length_pix
        pt_1 = (int(round(pos[0] - axis_x_view)), int(round(pos[1] - axis_y_view)))
        pt_2 = (int(round(pos[0] + axis_x_view)), int(round(pos[1] + axis_y_view)))
        cv.line(image, pt_1, pt_2, (255, 0, 0), 1)

        angle_rad = nmp.deg2rad(angle + 90)
        axis_x_view = nmp.cos(angle_rad) * axis_length_pix
        axis_y_view = nmp.sin(angle_rad) * axis_length_pix
        pt_1 = (int(round(pos[0] - axis_x_view)), int(round(pos[1] - axis_y_view)))
        pt_2 = (int(round(pos[0] + axis_x_view)), int(round(pos[1] + axis_y_view)))
        cv.line(image, pt_1, pt_2, (255, 0, 0), 1)

    def get_image_for_classification(self, contour, image):
        """Получение нормированного изображения для распознавания типа детали"""
        new_image = nmp.ones([self.image_size_for_storage[1], self.image_size_for_storage[0], 3])
        filling_color = nmp.mean(nmp.mean(self.img_orig_storage[0], axis=0), axis=0)
        new_image = (new_image * filling_color).astype(nmp.uint8)

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        contour_mask = nmp.zeros([self.image_size_for_storage[1], self.image_size_for_storage[0]])
        contour_mask = cv.fillPoly(contour_mask, [contour], 1).astype(nmp.bool)
        # contour_mask = cv.dilate(contour_mask, kernel).astype(nmp.bool)

        (x_crnter, y_center), _ = cv.minEnclosingCircle(contour)
        x_roll_mask, y_roll_mask = x_crnter - new_image.shape[1] // 2, y_center - new_image.shape[0] // 2
        x_roll_mask, y_roll_mask = int(x_roll_mask), int(y_roll_mask)
        dest_mask = nmp.roll(contour_mask, (-x_roll_mask, -y_roll_mask), (1, 0))

        new_image[dest_mask] = image[contour_mask]

        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)

        new_image = cv.GaussianBlur(new_image, (5, 5), 0.5)

        # plt.imshow(new_image)
        # plt.show()

        return new_image
