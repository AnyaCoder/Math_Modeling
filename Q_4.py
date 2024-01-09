
from utils import _sin, print_list, _cos, _tan, miles, three_cos_theory, center_elements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 定义读取的起始和结束列
start_col = 'C'
end_col = 'GU'

# 定义读取的起始和结束行
start_row = 3
end_row = 253

# 使用`usecols`参数来定义读取的列范围
cols = f"{start_col}:{end_col}"

# 读取Excel文件中的指定范围数据
df = pd.read_excel("附件.xlsx", engine='openpyxl', usecols=cols, header=None, skiprows=range(start_row - 1))

# 为了只读取到end_row行，需要从读取的数据中截取相应的行
df = df.iloc[:end_row - start_row + 1]

# 显示数据
print(df)

# 将深度变为负值
depth_values = -df.values

print(center_elements(depth_values))


def bilinear_interpolation(x, y, values):
    """双线性插值函数。
    values: 列表，包含四个点的深度值，按左上、右上、左下、右下的顺序。
    """
    q11, q21, q12, q22 = values
    return q11 * (1 - x) * (1 - y) + q21 * x * (1 - y) + q12 * (1 - x) * y + q22 * x * y


def find_intersections(depth_matrix, center_x_meters, center_y_meters, center_z_meters, step_size=1, cell_size=1.0):
    matrix_x = center_x_meters / cell_size
    matrix_y = center_y_meters / cell_size
    matrix_z = center_z_meters / cell_size
    max_range = max(depth_matrix.shape) * 2 * cell_size
    intersections = []
    matrix_x_size = depth_matrix.shape[1]
    matrix_y_size = depth_matrix.shape[0]
    for angle in np.linspace(30, 150, 3):

        dy = step_size * np.cos(np.radians(angle))
        dz = step_size * np.sin(np.radians(angle))

        y_offset, z = 0, matrix_z

        for _ in range(int(max_range / step_size)):
            y_offset += dy
            z -= dz
            # 在矩阵的坐标
            x_index = matrix_x
            y_index = matrix_y + y_offset / cell_size

            ix, iy = int(x_index), int(y_index)

            if ix < matrix_x_size - 1 and iy < matrix_y_size - 1:
                # 用双线性插值计算深度
                interpolated_depth = bilinear_interpolation(x_index - ix, y_index - iy,
                                                            [depth_matrix[iy, ix], depth_matrix[iy, ix + 1],
                                                             depth_matrix[iy + 1, ix], depth_matrix[iy + 1, ix + 1]])
                if z <= interpolated_depth:
                    intersections.append((x_index * cell_size, y_index * cell_size, z))
                    break
            elif ix < matrix_x_size - 1:
                # print('iy: ', (matrix_y_size - 1)  * cell_size)
                intersections.append((x_index * cell_size, (matrix_y_size - 1)  * cell_size, z))
                break
            elif iy < matrix_y_size- 1:
                # print('ix: ', (matrix_x_size - 1)  * cell_size)
                intersections.append(((matrix_y_size - 1)  * cell_size, y_index * cell_size, z))
                break
            else:
                # print('ix, iy: ', ((matrix_x_size- 1)  * cell_size, (matrix_y_size - 1) * cell_size))
                intersections.append(((matrix_x_size - 1)  * cell_size, (matrix_y_size - 1) * cell_size, z))
                break
    return intersections


def picture_draw(intersections, center_x_meters, center_y_meters):
    # 创建x, y坐标网格
    x = np.linspace(0, df.shape[1] - 1, df.shape[1]) * 0.02
    y = np.linspace(0, df.shape[0] - 1, df.shape[0]) * 0.02
    x, y = np.meshgrid(x, y)

    # 创建3D图形
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 添加交点和线段
    x_intersections = [pt[0] / 1852 for pt in intersections]
    y_intersections = [pt[1] / 1852 for pt in intersections]
    z_intersections = [pt[2] for pt in intersections]
    # 标注交点坐标
    # for xi, yi, zi in zip(x_intersections, y_intersections, z_intersections):
    #     label = '({:.2f}NM, {:.2f}NM, {:.2f}m)'.format(xi, yi, zi)
    #     ax.text(xi, yi, zi, label, color='black', zorder=6)  # Adjusting z-position slightly higher for visibility

    print(x_intersections)
    print(y_intersections)
    print(z_intersections)
    # 先画散点图
    ax.scatter(x_intersections, y_intersections, z_intersections, color='red', s=300, label='Intersections')
    if len(x_intersections) > 1:
        ax.plot(x_intersections, y_intersections, z_intersections, color='brown', label='Sonar Line')
     # 绘制海底深度图
    surf = ax.plot_surface(x, y, depth_values, cmap='viridis', linewidth=0, antialiased=False)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # 设置标题和轴标签
    ax.set_title('3D Sea Depth')
    ax.set_xlabel('Distance East (NM)')
    ax.set_ylabel('Distance South (NM)')
    ax.set_zlabel('Depth (m)')

    # 显示图形
    plt.show()


# 调用部分
cell_size = 0.02 * 1852
length_meters = 4 * miles
width_meters = 5 * miles
center_x_meters = (depth_values.shape[1] // 2) * cell_size
center_y_meters = (depth_values.shape[0] // 2) * cell_size
all_intersections = []

"""超参数"""
ita = -0.05   # 重叠率η
delta_x = 0.02  # 步长（海里)
eps = 1e-4
length_step = int(4 / delta_x)
""""""
result_line = []
result_matrix = []

min_y = 0.0
cycle = 0
while min_y < width_meters:
# while cycle < 40:
    print(f'cycle:{cycle}')
    x_idx = 0
    intended_y_list = []
    max_y_list = []
    for x_meter in np.linspace(0, length_meters - 1, length_step):
        # print('x_idx: ', x_idx)
        old_intersects = result_matrix[-1][x_idx] if cycle > 0 else [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        y_upper = old_intersects[0][1]   # 靠近上面的y
        y_lower = old_intersects[2][1]   # 靠近下面的y
        W_last = y_upper - y_lower # 上一行的覆盖宽度
        max_y_list.append(y_upper)
        intended_y = W_last * (1 - ita) + y_lower  # 这一条带的靠近下面的y所在的位置
        intended_y_list.append(intended_y)
        l, r = 0, -np.min(depth_values)
        new_intersects = None
         # 二分，找到覆盖的点
        while r - l > eps:
            mid = (l + r) / 2
            new_intersects = find_intersections(depth_matrix=depth_values,
                                            center_x_meters=x_meter,
                                            center_y_meters=intended_y + mid, # 靠近下面的y值+伸长的长度
                                            center_z_meters=0,
                                            cell_size=cell_size)
            new_y_lower = new_intersects[2][1]
            # print(intersects)
            if new_y_lower < intended_y:
                l = mid
            else:
                r = mid

        x_idx += 1
        # print(f'cycle:{cycle}, ', intersects)
        result_line.append(new_intersects)
    result_matrix.append(result_line)
    # print(result_matrix[-1][0], result_matrix[-1][-1])
    # 使用min()函数和lambda函数找到最小的y值
    min_y = min(max_y_list)
    print("最小的y值:", min_y)
    result_line = []
    cycle += 1


plt.figure(figsize=(10, 7))
# 设置标题和轴标签
plt.title("2D Plane of Points")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

# 设置轴长度
plt.xlim(0, 4 * 1852)
plt.ylim(0, 5 * 1852)
for cyc_cnt in range(cycle):
    # 提取 x 和 y 坐标
    x_coords_left = [pt[0][0] for pt in result_matrix[cyc_cnt]]
    y_coords_left = [pt[0][1] for pt in result_matrix[cyc_cnt]]
    x_coords_mid = [pt[1][0] for pt in result_matrix[cyc_cnt]]
    y_coords_mid = [pt[1][1] for pt in result_matrix[cyc_cnt]]
    x_coords_right = [pt[2][0] for pt in result_matrix[cyc_cnt]]
    y_coords_right = [pt[2][1] for pt in result_matrix[cyc_cnt]]
    # 创建图并绘制
    print(f'cyc_cnt: {cyc_cnt}, ')
    for i in range(len(x_coords_right)):
        if y_coords_left[i] < width_meters:
            plt.plot([x_coords_left[i], x_coords_right[i]], [y_coords_left[i], y_coords_right[i]]
                 , color='red' if cyc_cnt % 2 == 0 else 'blue', linewidth=2, alpha=0.1)
    for i in range(1, len(x_coords_mid)):
        if y_coords_left[i] < width_meters:
            plt.plot([x_coords_mid[i - 1], x_coords_mid[i]], [y_coords_mid[i - 1], y_coords_mid[i]]
                     , color='green', linewidth=1, alpha=1)
    # plt.scatter(x_coords_left, y_coords_left, color='red', s=0.1)  # 散点图
    # plt.scatter(x_coords_right, y_coords_right, color='red', s=0.1)  # 散点图

# 显示图
plt.grid(True)
plt.show()

from matplotlib import pyplot as plt
import numpy as np

plt.figure(figsize=(10, 7))
# 设置标题和轴标签
plt.title('问题3测线设计图', fontproperties="SimHei")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

x_lim = 4 * 1852
y_lim = 2 * 1852
# 设置轴长度
plt.xlim(0, x_lim)
plt.ylim(0, y_lim)

val = df.values
print(val[0, 1])
for i in range(cycle):
    plt.plot([val[i, 0] , val[i, 0]], [0, y_lim]
             , color='red' if i % 2 == 0 else 'blue', linewidth=1, alpha=1)
    for j in np.linspace(0, y_lim, 200):
        plt.plot([val[i, 2], val[i, 3]], [[j, j], [j, j]],
                 color='red' if i % 2 == 0 else 'blue', linewidth=2, alpha=0.05)
    # plt.scatter(x_coords_left, y_coords_left, color='red', s=0.1)  # 散点图
    # plt.scatter(x_coords_right, y_coords_right, color='red', s=0.1)  # 散点图

# 显示图
plt.grid(False)
plt.show()