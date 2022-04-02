import cv2
import numpy as np
import open3d as o3d


def convert_from_plane_to_3d(u, v, depth, cx, cy, focal_x, focal_y):
    x_over_z = (cx - u) / focal_x
    y_over_z = (cy - v) / focal_y

    z_matrix = depth / 5000

    x_matrix = x_over_z * z_matrix
    y_matrix = y_over_z * z_matrix
    return x_matrix, y_matrix, z_matrix


def main():
    color_matrix = cv2.imread('annot.png', cv2.IMREAD_COLOR)
    matrix_depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)

    print(color_matrix)

    rows, columns, channels = color_matrix.shape
    color_matrix_flattened = color_matrix.reshape(rows * columns, 3)

    columns_indices = np.arange(0, columns)

    matrix_v = np.tile(columns_indices, (rows, 1))
    matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

    x, y, z, = convert_from_plane_to_3d(matrix_u, matrix_v, matrix_depth, cx=319.50, cy=239.50, focal_x=481.20,
                                        focal_y=-480.00)  # getting xyz coordinates of each point

    matrix_xyz = np.dstack((x, y, z))
    answ_matrix = matrix_xyz.reshape(rows * columns, 3)  # now we have a list of points

    pc = o3d.geometry.PointCloud()

    pc.points = o3d.utility.Vector3dVector(answ_matrix)
    pc.colors = o3d.utility.Vector3dVector(color_matrix_flattened.astype(np.float64) / 255.0)

    if pc.has_colors():
        colors = np.asarray(pc.colors)
    elif pc.has_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(pc.normals) * 0.5
    else:
        pc.paint_uniform_color((1.0, 0.0, 0.0))
        colors = np.asarray(pc.colors)

    o3d.visualization.draw_geometries([pc])


if __name__ == '__main__':
    main()
