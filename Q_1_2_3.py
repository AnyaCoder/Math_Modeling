# encoding: utf-8

import math


class TestLine:
    def __init__(self, distance_to_center):
        super()
        # 距离中心的距离
        self.distance_to_center = distance_to_center
        # 海水深度
        self.height = 0
        # 覆盖宽度系数
        self.k = 0
        # 覆盖宽度
        self.w = 0


"""
------------------------------math tools-----------------------------
"""


def tan(angle_degrees):
    # 将角度转换为弧度
    angle_radians = math.radians(angle_degrees)

    # 计算正切值
    tan_value = math.tan(angle_radians)

    return tan_value


def cos(angle_degrees):
    # 将角度转换为弧度
    angle_radians = math.radians(angle_degrees)

    # 计算正切值
    tan_value = math.cos(angle_radians)

    return tan_value


def K(theta_degrees, alpha_degrees):
    # 将角度转换为弧度
    theta = math.radians(theta_degrees)
    alpha = math.radians(alpha_degrees)

    # 计算公式中的各部分
    part1 = 1 / math.cos(theta / 2 + alpha)
    part2 = 1 / math.cos(theta / 2 - alpha)
    part3 = math.sin(theta / 2)

    # 计算 k 的值
    res = (part1 + part2) * part3 * cos(alpha_degrees)

    return res


def get_gamma(beta_degrees, alpha_degrees):
    """
    计算波束所在的平面对应的坡角
    :param beta_degrees: 航向角
    :param alpha_degrees: 矩形海域的坡度角
    :return: 计算出的坡角
    """
    # 将角度转换为弧度
    beta = math.radians(beta_degrees)
    alpha = math.radians(alpha_degrees)

    # 计算公式的值
    gamma = math.atan(-math.sin(beta) * math.tan(alpha))

    # 将弧度转换为度数
    gamma_degrees = math.degrees(gamma)

    return gamma_degrees


def get_delta(beta_degrees, alpha_degrees):
    """
    计算测量船前进的坡度角
    :param beta_degrees: 航向角
    :param alpha_degrees: 矩形海域的坡度角
    :return: 计算出的对应的角度
    """
    # 将角度转换为弧度
    beta = math.radians(beta_degrees)
    alpha = math.radians(alpha_degrees)

    # 计算公式的值
    delta = math.atan(-math.cos(beta) * math.tan(alpha))

    # 将弧度转换为度数
    delta_degrees = math.degrees(delta)

    return delta_degrees


def higher_k(theta_degrees, alpha_degrees):
    # 将角度转换为弧度
    theta = math.radians(theta_degrees)
    alpha = math.radians(alpha_degrees)

    # 计算公式的值
    result = math.sin(theta / 2) / math.cos(theta / 2 + alpha) * cos(alpha_degrees)

    return result


def lower_k(theta_degrees, alpha_degrees):
    # 将角度转换为弧度
    theta = math.radians(theta_degrees)
    alpha = math.radians(alpha_degrees)

    # 计算公式的值
    result = math.sin(theta / 2) / math.cos(theta / 2 - alpha) * cos(alpha_degrees)

    return result


"""
---------------------------------------------------------------------
"""


def problem1_solution():
    test_line_list = [TestLine(i) for i in range(-800, 801, 200)]

    # 初始赋值
    test_line_list[4].height = 70

    # 计算海水深度
    for i in range(-4, 5):
        test_line_list[4 + i].height = test_line_list[4].height + (
                test_line_list[4].distance_to_center - test_line_list[4 + i].distance_to_center) * tan(1.5)

    # 计算覆盖宽度
    for i in range(len(test_line_list)):
        test_line_list[i].k = K(120, 1.5)
        test_line_list[i].w = test_line_list[i].k * test_line_list[i].height

    overlap_ratio_list = [-1]

    # 计算与前一条覆盖宽度的重叠率
    for i in range(1, len(test_line_list)):
        t1 = test_line_list[i - 1]
        t2 = test_line_list[i]

        overlap_ratio = t1.height * t1.k + t2.height * t2.k - t1.height * higher_k(120, 1.5) - t2.height * lower_k(120,
                                                                                                                   1.5) - (
                                t2.distance_to_center - t1.distance_to_center)

        overlap_ratio_list.append(overlap_ratio / t1.w)

    print('距离中心的距离：')
    for item in test_line_list:
        print(item.distance_to_center)

    print('海水深度：')
    for item in test_line_list:
        print(item.height)

    print('覆盖宽度：')
    for item in test_line_list:
        print(item.w)

    print('与上一条测线的重叠率：')
    for item in overlap_ratio_list:
        print(item)


def problem2_solution():
    test_line_list = [TestLine(i / 10 * 1852) for i in range(0, 22, 3)]
    # for item in test_line_list:
    #     print(item.distance_to_center)

    test_line_list[0].height = 120

    for beta_degrees in range(0, 320, 45):
        # 获取波束新的坡度角
        gamma_degrees = get_gamma(beta_degrees, 1.5)

        # 获取前进坡面的坡度角
        delta_degrees = get_delta(beta_degrees, 1.5)

        # 计算海水深度
        for i in range(1, 8):
            test_line_list[i].height = test_line_list[0].height + (
                    test_line_list[0].distance_to_center - test_line_list[i].distance_to_center) * tan(
                delta_degrees)

        # 计算覆盖宽度
        for i in range(len(test_line_list)):
            test_line_list[i].k = K(120, gamma_degrees)
            test_line_list[i].w = test_line_list[i].k * test_line_list[i].height

        print(beta_degrees, ":")

        print("海水深度：")
        for item in test_line_list:
            print(item.height)

        for item in test_line_list:
            print(item.w, end='   ')
        print()


def problem3_solution():
    theta_degrees = 120
    alpha_degrees = 1.5
    L = 4 * 1852
    r = tan(alpha_degrees)
    h = 110 + L / 2 * r
    k = K(theta_degrees, alpha_degrees)
    k_left = higher_k(theta_degrees, alpha_degrees)
    k_right = lower_k(theta_degrees, alpha_degrees)
    eta = 0.1

    x_list = list()

    # 首项
    x = h * k_left / (1 + r * k_left)
    x_list.append(x)

    p1 = r * k_right - 1 - r * k
    p2 = r * k - 1 - r * k_left - eta * r * k
    p3 = h * (k_left + k_right) + (eta - 2) * h * k

    while True:
        x = x_list[-1]
        y = p2 / p1 * x + p3 / p1
        x_list.append(y)
        bound = (L - h * k_right) / (1 - r * k_right)
        if y >= bound:
            break

    for i in x_list:
        print(i)
        print(i - (h - r * i) * k_left, i + (h - r * i) * k_right)

    print(len(x_list))


if __name__ == '__main__':
    problem3_solution()
