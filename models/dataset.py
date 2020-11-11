# pylint: disable=E1101,R,C
import csv
import ctypes
import errno
import glob
import os
import re
import threading

import numpy as np
import torch
import torch.utils.data
import trimesh
import logging
import random
from sklearn.linear_model import LinearRegression
import time

logging.getLogger('pyembree').disabled = True


def load_obj(path):
    # import os
    # import numpy as np
    with open(path) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
    # points原本为列表，需要转变为矩阵，方便处理
    points = np.array(points)
    return points


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


# b refers to bandwidth
def make_sgrid(b, alpha, beta, gamma):
    from lie_learn.spaces import S2

    beta, alpha = S2.meshgrid(b=b, grid_type='SOFT')

    sgrid = S2.change_coordinates(np.c_[beta[..., None], alpha[..., None]], p_from='S', p_to='C')

    # sgrid = sgrid.reshape((-1, 3))

    # R = rotmat(alpha, beta, gamma, hom_coord=False)
    # sgrid = np.einsum('ij,nj->ni', R, sgrid)#rotation

    return sgrid


def render_model(mesh, sgrid):
    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))  # fix bug if loc is empty

    # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
    grid_hits = sgrid[index_ray]
    grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

    # Compute the distance from the grid points to the intersection pionts
    dist = np.linalg.norm(grid_hits - loc, axis=-1)

    # For each intersection, look up the normal of the triangle that was hit
    normals = mesh.face_normals[index_tri]
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Construct spherical images
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    # dist_im = dist_im.reshape(theta.shape)

    # shaded_im = np.zeros(sgrid.shape[0])
    # shaded_im[index_ray] = normals.dot(light_dir)
    # shaded_im = shaded_im.reshape(theta.shape) + 0.4

    n_dot_ray_im = np.zeros(sgrid.shape[0])
    # n_dot_ray_im[index_ray] = np.abs(np.einsum("ij,ij->i", normals, grid_hits_normalized))
    n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals, grid_hits_normalized)

    nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
    gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], grid_hits_normalized[:, 2]
    wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2)
    n_wedge_ray_im = np.zeros(sgrid.shape[0])
    n_wedge_ray_im[index_ray] = wedge_norm

    # Combine channels to construct final image
    # im = dist_im.reshape((1,) + dist_im.shape)
    im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)

    return im


def linear_regression(northwest: np.ndarray = None, north: np.ndarray = None, northeast: np.ndarray = None,
                      west: np.ndarray = None, center: np.ndarray = None, east: np.ndarray = None,
                      southwest: np.ndarray = None, south: np.ndarray = None, southeast: np.ndarray = None, lib=None):
    lr = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    coef = np.zeros((center.shape[0], 2))
    intercept = np.zeros((center.shape[0], 1))
    points = center
    if northwest is not None:
        points = np.concatenate((points, northwest), axis=0)
    if north is not None:
        points = np.concatenate((points, north), axis=0)
    if northeast is not None:
        points = np.concatenate((points, northeast), axis=0)
    if west is not None:
        points = np.concatenate((points, west), axis=0)
    if east is not None:
        points = np.concatenate((points, east), axis=0)
    if southwest is not None:
        points = np.concatenate((points, southwest), axis=0)
    if south is not None:
        points = np.concatenate((points, south), axis=0)
    if southeast is not None:
        points = np.concatenate((points, southeast), axis=0)
    points = points.reshape(-1, center.shape[0], center.shape[1])
    points = np.transpose(points, (1, 0, 2))
    # def thread(start,stop):

    """
    for index in range(center.shape[0]):
    #points = np.stack((northwest[index], north[index], northeast[index], west[index], center[index], east[index], southwest[index], south[index], southeast[index]), axis=0)
    X = points[index, :, 0:2]
    Y = points[index, :, 2]
    reg = lr.fit(X, Y)
    # coef = reg.coef_
    coef[index] = reg.coef_
    # intercept = reg.intercept_
    intercept[index] = reg.intercept_
    """

    if lib is None:
        _file = 'libmatrix.so'
        _path = '/data2/tzf/s2cnn-master/draft/' + _file
        lib = ctypes.cdll.LoadLibrary(_path)
    points = points.astype(np.float32)
    m = np.zeros((center.shape[0], 3))
    m = m.astype(np.float32)

    c_test = lib._linear_regression
    c_test.restype = None
    c_test.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=3),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2),
    ]

    c_test(points, np.array(points.shape).astype(np.int32), m)
    intercept[:, 0] = m[:, 2]
    coef = m[:, 0:2]
    """
    
    """

    return coef, intercept


def interpolate(m: np.ndarray, n: np.ndarray, sgrid: np.ndarray, points_on_sphere: np.ndarray, radius: np.ndarray):
    # print("Interpolate")
    """
    m = m.copy()
    n = n.copy()
    sgrid = sgrid.copy()
    points_on_sphere = points_on_sphere.copy()
    radius = radius.copy()

    """

    # =========================
    center_grid = np.zeros((m.shape[0], 3))
    east_grid = np.zeros((m.shape[0], 3))
    south_grid = np.zeros((m.shape[0], 3))
    southeast_grid = np.zeros((m.shape[0], 3))

    center_dist = np.zeros(m.shape[0])
    east_dist = np.zeros(m.shape[0])
    south_dist = np.zeros(m.shape[0])
    southeast_dist = np.zeros(m.shape[0])

    center_weight = np.zeros(m.shape[0])
    east_weight = np.zeros(m.shape[0])
    south_weight = np.zeros(m.shape[0])
    southeast_weight = np.zeros(m.shape[0])

    # center_index = np.zeros((m.shape[0], 2))
    # east_index = np.zeros((m.shape[0], 2))
    # south_index = np.zeros((m.shape[0], 2))
    # southeast_index = np.zeros((m.shape[0], 2))

    # use a mask to select the point on the  boundary============================
    mask_north = m == 0
    mask_south = m == sgrid.shape[0] - 1
    mask_boundary = mask_north + mask_south
    m_boundary = m[mask_boundary]
    n_boundary = n[mask_boundary] % sgrid.shape[1]
    n_boundary_plus_one = (n_boundary + 1) % sgrid.shape[1]
    n_boundary_opposite = (n_boundary + (sgrid.shape[1] / 2)) % sgrid.shape[1]
    n_boundary_opposite = n_boundary_opposite.astype(int)
    n_boundary_plus_one_opposite = (n_boundary_plus_one + (sgrid.shape[1] / 2)) % sgrid.shape[1]
    n_boundary_plus_one_opposite = n_boundary_plus_one_opposite.astype(int)

    center_grid[mask_boundary] = sgrid[m_boundary, n_boundary]
    east_grid[mask_boundary] = sgrid[m_boundary, n_boundary_plus_one]
    south_grid[mask_boundary] = sgrid[m_boundary, n_boundary_opposite]
    southeast_grid[mask_boundary] = sgrid[m_boundary, n_boundary_plus_one_opposite]

    # center_index[mask_boundary] = np.array((m_boundary, n_boundary)).transpose(1, 0)
    # east_index[mask_boundary] = np.array((m_boundary, n_boundary_plus_one)).transpose(1, 0)
    # south_index[mask_boundary] = np.array((m_boundary, n_boundary_opposite)).transpose(1, 0)
    # southeast_index[mask_boundary] = np.array((m_boundary, n_boundary_plus_one_opposite)).transpose(1, 0)

    # calculate distance and relevant weight
    center_dist[mask_boundary] = np.sqrt(np.sum((center_grid[mask_boundary] - points_on_sphere[mask_boundary]) ** 2))
    east_dist[mask_boundary] = np.sqrt(np.sum((east_grid[mask_boundary] - points_on_sphere[mask_boundary]) ** 2))
    south_dist[mask_boundary] = np.sqrt(np.sum((south_grid[mask_boundary] - points_on_sphere[mask_boundary]) ** 2))
    southeast_dist[mask_boundary] = np.sqrt(
        np.sum((southeast_grid[mask_boundary] - points_on_sphere[mask_boundary]) ** 2))
    sum = center_dist[mask_boundary] + east_dist[mask_boundary] + south_dist[mask_boundary] + southeast_dist[
        mask_boundary]
    center_weight[mask_boundary] = center_dist[mask_boundary] / sum
    east_weight[mask_boundary] = east_dist[mask_boundary] / sum
    south_weight[mask_boundary] = south_dist[mask_boundary] / sum
    southeast_weight[mask_boundary] = southeast_dist[mask_boundary] / sum

    # save the signal of distance
    radius_boundary = radius[mask_boundary]
    dist_im = np.zeros(sgrid.shape[0:2])  # signal of distance from points to sphere
    weight_im = np.zeros(sgrid.shape[
                         0:2])  # Since each grid point on the sphere could be affected by several different signals, we need to normalize the values.
    dist_im[m_boundary, n_boundary] += radius_boundary[:, 0] * center_weight[mask_boundary]
    dist_im[m_boundary, n_boundary_plus_one] += radius_boundary[:, 0] * east_weight[mask_boundary]
    dist_im[m_boundary, n_boundary_opposite] += radius_boundary[:, 0] * south_weight[mask_boundary]
    dist_im[m_boundary, n_boundary_plus_one_opposite] += radius_boundary[:, 0] * southeast_weight[mask_boundary]
    weight_im[m_boundary, n_boundary] += center_weight[mask_boundary]
    weight_im[m_boundary, n_boundary_plus_one] += east_weight[mask_boundary]
    weight_im[m_boundary, n_boundary_opposite] += south_weight[mask_boundary]
    weight_im[m_boundary, n_boundary_plus_one_opposite] += southeast_weight[mask_boundary]

    # use a mask to select the rest points===============================
    mask_rest = ~mask_boundary
    m_rest = m[mask_rest]
    n_rest = n[mask_rest] % sgrid.shape[1]
    n_rest_plus_one = (n_rest + 1) % sgrid.shape[1]

    center_grid[mask_rest] = sgrid[m_rest, n_rest]
    east_grid[mask_rest] = sgrid[m_rest, n_rest_plus_one]
    south_grid[mask_rest] = sgrid[m_rest + 1, n_rest]
    southeast_grid[mask_rest] = sgrid[m_rest + 1, n_rest_plus_one]

    # center_index[mask_rest] = np.array((m_rest, n_rest)).transpose(1, 0)
    # east_index[mask_rest] = np.array((m_rest, n_rest_plus_one)).transpose(1, 0)
    # south_index[mask_rest] = np.array((m_rest + 1, n_rest)).transpose(1, 0)
    # southeast_index[mask_rest] = np.array((m_rest + 1, n_rest_plus_one)).transpose(1, 0)

    # calculate distance and relevant weight
    center_dist[mask_rest] = np.sqrt(np.sum((center_grid[mask_rest] - points_on_sphere[mask_rest]) ** 2))
    east_dist[mask_rest] = np.sqrt(np.sum((east_grid[mask_rest] - points_on_sphere[mask_rest]) ** 2))
    south_dist[mask_rest] = np.sqrt(np.sum((south_grid[mask_rest] - points_on_sphere[mask_rest]) ** 2))
    southeast_dist[mask_rest] = np.sqrt(np.sum((southeast_grid[mask_rest] - points_on_sphere[mask_rest]) ** 2))
    sum = center_dist[mask_rest] + east_dist[mask_rest] + south_dist[mask_rest] + southeast_dist[mask_rest]
    center_weight[mask_rest] = center_dist[mask_rest] / sum
    east_weight[mask_rest] = east_dist[mask_rest] / sum
    south_weight[mask_rest] = south_dist[mask_rest] / sum
    southeast_weight[mask_rest] = southeast_dist[mask_rest] / sum

    # save the signal of distance
    radius_rest = radius[mask_rest]
    dist_im = np.zeros(sgrid.shape[0:2])  # signal of distance from points to sphere
    weight_im = np.zeros(sgrid.shape[
                         0:2])  # Since each grid point on the sphere could be affected by several different signals, we need to normalize the values.
    dist_im[m_rest, n_rest] += radius_rest[:, 0] * center_weight[mask_rest]
    dist_im[m_rest, n_rest_plus_one] += radius_rest[:, 0] * east_weight[mask_rest]
    dist_im[m_rest + 1, n_rest] += radius_rest[:, 0] * south_weight[mask_rest]
    dist_im[m_rest + 1, n_rest_plus_one] += radius_rest[:, 0] * southeast_weight[mask_rest]
    weight_im[m_rest, n_rest] += center_weight[mask_rest]
    weight_im[m_rest, n_rest_plus_one] += east_weight[mask_rest]
    weight_im[m_rest + 1, n_rest] += south_weight[mask_rest]
    weight_im[m_rest + 1, n_rest_plus_one] += southeast_weight[mask_rest]

    mask_weight = weight_im != 0
    dist_im[mask_weight] /= weight_im[mask_weight]
    dist_im = 1 - dist_im
    return dist_im


def angle(m: np.ndarray, n: np.ndarray, sgrid: np.ndarray, dist_im: np.ndarray, lib):
    # print("angle")
    """
    m=m.copy()
    n=n.copy()

    sgrid=sgrid.copy()
    dist_im=dist_im.copy()

    """
    n = n % sgrid.shape[1]
    # utilizing linear regression to create a plane

    # use a mask to avoid the index out of the boundary
    """
    mask_m = m - 1 >= 0
    mask = mask_m
    m = m[mask]
    n = n[mask]
    """
    time_before_initialization = time.perf_counter()
    # Initialize the variables
    northwest_points = np.zeros((m.shape[0], 3))
    north_points = np.zeros((m.shape[0], 3))
    northeast_points = np.zeros((m.shape[0], 3))
    west_points = np.zeros((m.shape[0], 3))
    center_points = np.zeros((m.shape[0], 3))
    east_points = np.zeros((m.shape[0], 3))
    southwest_points = np.zeros((m.shape[0], 3))
    south_points = np.zeros((m.shape[0], 3))
    southeast_points = np.zeros((m.shape[0], 3))
    coef = np.zeros((m.shape[0], 2))
    intercept = np.zeros((m.shape[0], 1))

    # calculate the coef & intercept of points on the north boundary
    time_before_north = time.perf_counter()
    # print("time_before_north=time.perf_counter()")
    # print(time_before_north-time_before_initialization)
    mask_north = m == 0
    m_north = m[mask_north]
    if m_north.size != 0:
        n_north = n[mask_north] % sgrid.shape[1]
        n_north_minus_one = (n_north - 1) % sgrid.shape[1]
        n_north_plus_one = (n_north + 1) % sgrid.shape[1]
        west_points[mask_north] = sgrid[m_north, n_north_minus_one] * (
                1 - np.repeat(dist_im[m_north, n_north_minus_one], 3).reshape(-1, 3))
        center_points[mask_north] = sgrid[m_north, n_north] * (
                1 - np.repeat(dist_im[m_north, n_north], 3).reshape(-1, 3))
        east_points[mask_north] = sgrid[m_north, n_north_plus_one] * (
                1 - np.repeat(dist_im[m_north, n_north_plus_one], 3).reshape(-1, 3))
        southwest_points[mask_north] = sgrid[m_north + 1, n_north_minus_one] * (
                1 - np.repeat(dist_im[m_north + 1, n_north_minus_one], 3).reshape(-1, 3))
        south_points[mask_north] = sgrid[m_north + 1, n_north] * (
                1 - np.repeat(dist_im[m_north + 1, n_north], 3).reshape(-1, 3))
        southeast_points[mask_north] = sgrid[m_north + 1, n_north_plus_one] * (
                1 - np.repeat(dist_im[m_north + 1, n_north_plus_one], 3).reshape(-1, 3))

        coef[mask_north, :], intercept[mask_north] = linear_regression(northwest=None, north=None, northeast=None,
                                                                       west=west_points[mask_north],
                                                                       center=center_points[mask_north],
                                                                       east=east_points[mask_north],
                                                                       southwest=southwest_points[mask_north],
                                                                       south=south_points[mask_north],
                                                                       southeast=southeast_points[mask_north], lib=lib)

    # calculate the coef & intercept of points on the south boundary
    time_before_south = time.perf_counter()
    # print("time_before_south-time_before_north")
    # print(time_before_south-time_before_north)
    mask_south = m == sgrid.shape[0] - 1
    m_south = m[mask_south]
    if m_south.size != 0:
        n_south = n[mask_south] % sgrid.shape[1]
        n_south_minus_one = (n_south - 1) % sgrid.shape[1]
        n_south_plus_one = (n_south + 1) % sgrid.shape[1]
        northwest_points[mask_south] = sgrid[m_south - 1, n_south_minus_one] * (
                1 - np.repeat(dist_im[m_south - 1, n_south_minus_one], 3).reshape(-1, 3))
        north_points[mask_south] = sgrid[m_south - 1, n_south] * (
                1 - np.repeat(dist_im[m_south - 1, n_south], 3).reshape(-1, 3))
        northeast_points[mask_south] = sgrid[m_south - 1, n_south_plus_one] * (
                1 - np.repeat(dist_im[m_south - 1, n_south_plus_one], 3).reshape(-1, 3))
        west_points[mask_south] = sgrid[m_south, n_south_minus_one] * (
                1 - np.repeat(dist_im[m_south, n_south_minus_one], 3).reshape(-1, 3))
        center_points[mask_south] = sgrid[m_south, n_south] * (
                1 - np.repeat(dist_im[m_south, n_south], 3).reshape(-1, 3))
        east_points[mask_south] = sgrid[m_south, n_south_plus_one] * (
                1 - np.repeat(dist_im[m_south, n_south_plus_one], 3).reshape(-1, 3))

        coef[mask_south, :], intercept[mask_south] = linear_regression(northwest=northwest_points[mask_south],
                                                                       north=north_points[mask_south],
                                                                       northeast=northeast_points[mask_south],
                                                                       west=west_points[mask_south],
                                                                       center=center_points[mask_south],
                                                                       east=east_points[mask_south],
                                                                       southwest=None, south=None, southeast=None,
                                                                       lib=lib)

    # calculate the rest points
    time_before_rest = time.perf_counter()
    mask_boundary = mask_north + mask_south
    mask_rest = ~mask_boundary
    m_rest = m[mask_rest]
    n_rest = n[mask_rest] % sgrid.shape[1]
    n_rest_minus_one = (n_rest - 1) % sgrid.shape[1]
    n_rest_plus_one = (n_rest + 1) % sgrid.shape[1]

    # calculate the estimated position of the points corresponded to the grids
    northwest_points[mask_rest] = sgrid[m_rest - 1, n_rest_minus_one] * (
            1 - np.repeat(dist_im[m_rest - 1, n_rest_minus_one], 3).reshape(-1, 3))
    north_points[mask_rest] = sgrid[m_rest - 1, n_rest] * (1 - np.repeat(dist_im[m_rest - 1, n_rest], 3).reshape(-1, 3))
    northeast_points[mask_rest] = sgrid[m_rest - 1, n_rest_plus_one] * (
            1 - np.repeat(dist_im[m_rest - 1, n_rest_plus_one], 3).reshape(-1, 3))
    west_points[mask_rest] = sgrid[m_rest, n_rest_minus_one] * (
            1 - np.repeat(dist_im[m_rest, n_rest_minus_one], 3).reshape(-1, 3))
    center_points[mask_rest] = sgrid[m_rest, n_rest] * (1 - np.repeat(dist_im[m_rest, n_rest], 3).reshape(-1, 3))
    east_points[mask_rest] = sgrid[m_rest, n_rest_plus_one] * (
            1 - np.repeat(dist_im[m_rest, n_rest_plus_one], 3).reshape(-1, 3))
    southwest_points[mask_rest] = sgrid[m_rest + 1, n_rest_minus_one] * (
            1 - np.repeat(dist_im[m_rest + 1, n_rest_minus_one], 3).reshape(-1, 3))
    south_points[mask_rest] = sgrid[m_rest + 1, n_rest] * (1 - np.repeat(dist_im[m_rest + 1, n_rest], 3).reshape(-1, 3))
    southeast_points[mask_rest] = sgrid[m_rest + 1, n_rest_plus_one] * (
            1 - np.repeat(dist_im[m_rest + 1, n_rest_plus_one], 3).reshape(-1, 3))
    time_before_lr = time.perf_counter()
    # print("time_before_lr - time_before_rest")
    # print(time_before_lr-time_before_rest)
    coef[mask_rest, :], intercept[mask_rest] = linear_regression(northwest=northwest_points[mask_rest],
                                                                 north=north_points[mask_rest],
                                                                 northeast=northeast_points[mask_rest],
                                                                 west=west_points[mask_rest],
                                                                 center=center_points[mask_rest],
                                                                 east=east_points[mask_rest],
                                                                 southwest=southwest_points[mask_rest],
                                                                 south=south_points[mask_rest],
                                                                 southeast=southeast_points[mask_rest], lib=lib)
    time_after_lr = time.perf_counter()
    # print("time_after_lr - time_before_lr")
    # print(time_after_lr - time_before_lr)
    # calculate the angle signals
    dot_im = np.zeros(sgrid.shape[0:2])  # signal of dot production between rays and normals
    cross_im = dist_im = np.zeros(sgrid.shape[0:2])  # signal of cross production between rays and normals
    normals = np.zeros((m.shape[0], 3))
    normals[:, 0:2] = coef
    normals[:, 2] = -1
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    dot_im[m, n] = np.abs(np.einsum("ij,ij->i", normalized_normals, sgrid[m, n]))
    nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
    cx, cy, cz = center_points[:, 0], center_points[:, 1], center_points[:, 2]
    cross_im[m, n] = np.sqrt((nx * cy - ny * cx) ** 2 + (nx * cz - nz * cx) ** 2 + (ny * cz - nz * cy) ** 2)
    time_end = time.perf_counter()
    # print("time_end - time_after_lr")
    # print(time_end - time_after_lr)
    return dot_im, cross_im
    # return 1,1


def inverse_render_model(points: np.ndarray, sgrid: np.ndarray, lib):
    # wait for implementing
    # print("InverseRenderModel")
    try:
        x = points[:, 0]
    except IndexError:
        print(points.shape[0], points.shape[1])

    y = points[:, 1]
    z = points[:, 2]
    # Fistly, we get the centroid, then translate the points
    centroid_x = np.sum(x) / points.shape[0]
    centroid_y = np.sum(y) / points.shape[0]
    centroid_z = np.sum(z) / points.shape[0]
    centroid = np.array([centroid_x, centroid_y, centroid_z])
    points = points.astype(np.float)
    points -= centroid

    # After normalization, compute the distance between the sphere and points

    radius = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
    # dist = 1 - (1 / np.max(radius)) * radius

    # Projection
    from lie_learn.spaces import S2
    radius = np.repeat(radius, 3).reshape(-1, 3)
    points_on_sphere = points / radius
    # ssgrid = sgrid.reshape(-1, 3)
    # phi, theta = S2.change_coordinates(ssgrid, p_from='C', p_to='S')
    out = S2.change_coordinates(points_on_sphere, p_from='C', p_to='S')

    phi = out[..., 0]
    theta = out[..., 1]

    phi = phi
    theta = theta % (np.pi * 2)

    # Interpolate
    b = sgrid.shape[0] / 2  # bandwidth
    # By computing the m,n, we can find
    # the neighbours on the sphere
    m = np.trunc((phi - np.pi / (4 * b)) / (np.pi / (2 * b)))
    m = m.astype(int)
    n = np.trunc(theta / (np.pi / b))
    n = n.astype(int)

    dist_im = interpolate(
        m=m, n=n, sgrid=sgrid,
        points_on_sphere=points_on_sphere,
        radius=radius)
    dot_img, cross_img = angle(m=m, n=n, sgrid=sgrid, dist_im=dist_im, lib=lib)
    # dist_im=dist_im.reshape(-1,1)

    # wait for validation
    im = np.stack((dist_im, dot_img, cross_img), axis=0)

    return im


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, True)
    return rot


class ToMesh:
    def __init__(self, random_rotations=False, random_translation=0):
        self.rot = random_rotations
        self.tr = random_translation

    def __call__(self, path):
        mesh = trimesh.load_mesh(path)
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        mesh.fill_holes()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        mesh.apply_translation(-mesh.centroid)

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(1 / r)

        if self.tr > 0:
            tr = np.random.rand() * self.tr
            rot = rnd_rot()
            mesh.apply_transform(rot)
            mesh.apply_translation([tr, 0, 0])

            if not self.rot:
                mesh.apply_transform(rot.T)

        if self.rot:
            mesh.apply_transform(rnd_rot())

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(0.99 / r)  # Normalization

        return mesh

    def __repr__(self):
        return self.__class__.__name__ + '(rotation={0}, translation={1})'.format(self.rot, self.tr)


class ToPoints:
    def __init__(self, random_rotations=False, random_translation=0):
        self.rot = random_rotations
        self.tr = random_translation

    def __call__(self, path):
        # print("ToPoints")
        # mesh=trimesh.load_mesh(path)
        # points=mesh.vertices.view(np.ndarray)
        points = load_obj(path=path)
        return points
        # mesh=trimesh.load_mesh(path)


class ProjectOnSphere:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.sgrid = make_sgrid(bandwidth, alpha=0, beta=0, gamma=0)  # create a sphere grid

    def __call__(self, mesh):
        im = render_model(mesh, self.sgrid)
        im = im.reshape(3, 2 * self.bandwidth, 2 * self.bandwidth)

        from scipy.spatial.qhull import QhullError  # pylint: disable=E0611
        try:
            convex_hull = mesh.convex_hull
        except QhullError:
            convex_hull = mesh
        # Also cast ray on the corresponding convex hull, generating a 6-channel features.
        # hull_im = render_model(convex_hull, self.sgrid)
        # hull_im = hull_im.reshape(3, 2 * self.bandwidth, 2 * self.bandwidth)

        # im = np.concatenate([im, hull_im], axis=0)
        assert len(im) == 6
        # random translation?
        im[0] -= 0.75
        im[0] /= 0.26
        im[1] -= 0.59
        im[1] /= 0.50
        im[2] -= 0.54
        im[2] /= 0.29
        im[3] -= 0.52
        im[3] /= 0.19
        im[4] -= 0.80
        im[4] /= 0.18
        im[5] -= 0.51
        im[5] /= 0.25

        im = im.astype(np.float32)  # pylint: disable=E1101

        return im

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)


class ProjectFromPointsOnSphere:
    # Wait for implementing
    def __init__(self, bandwidth, lib):
        self.bandwidth = bandwidth
        self.sgrid = make_sgrid(bandwidth, alpha=0, beta=0, gamma=0)  # create a sphere grid
        self.lib = lib

    def __call__(self, points, length):
        # print("ProjectFromPointsOnSPhere")
        # the sum of length should be equal to the counts of points
        assert sum(length) == points.shape[0]
        im_list = []
        cur_index = 0
        for index in range(len(length)):
            im = inverse_render_model(
                points[cur_index:cur_index + length[index], ...], self.sgrid, self.lib)
            im = im.astype(np.float32)
            im_list.append(im)
            cur_index += length[index]
        return im_list

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)


class CacheNPY:
    def __init__(self, prefix, repeat, transform, pick_randomly=True):
        self.transform = transform
        self.prefix = prefix
        self.repeat = repeat
        self.pick_randomly = pick_randomly

    def check_trans(self, file_path):
        # print("transform {}...".format(file_path))
        try:
            return self.transform(file_path)
        except:
            print("Exception during transform of {}".format(file_path))
            raise

    def __call__(self, file_path):
        head, tail = os.path.split(file_path)
        root, _ = os.path.splitext(tail)
        npy_path = os.path.join(head, self.prefix + root + '_{0}.npy')

        exists = [os.path.exists(npy_path.format(i)) for i in range(self.repeat)]

        if self.pick_randomly and all(exists):
            i = np.random.randint(self.repeat)
            try:
                return np.load(npy_path.format(i))
            except OSError:
                exists[i] = False

        if self.pick_randomly:
            img = self.check_trans(file_path)
            np.save(npy_path.format(exists.index(False)), img)

            return img

        output = []
        for i in range(self.repeat):
            try:
                img = np.load(npy_path.format(i))
            except (OSError, FileNotFoundError):
                img = self.check_trans(file_path)
                np.save(npy_path.format(i), img)
            output.append(img)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(prefix={0}, transform={1})'.format(self.prefix, self.transform)


class Shrec17(torch.utils.data.Dataset):
    '''
    Download SHREC17 and output valid obj files content
    '''

    url_data = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.zip'
    url_label = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.csv'

    def __init__(self, root, dataset, perturbed=True, download=False, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)

        if not dataset in ["train", "test", "val"]:
            raise ValueError("Invalid dataset")

        self.dir = os.path.join(self.root, dataset + ("_perturbed" if perturbed else ""))
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download(dataset, perturbed)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.files = sorted(glob.glob(os.path.join(self.dir, '*/*.obj')))
        if dataset != "test":
            with open(os.path.join(self.root, dataset + ".csv"), 'rt') as f:
                reader = csv.reader(f)
                self.labels = {}
                for row in [x for x in reader][1:]:
                    self.labels[row[0]] = (row[1], row[2])
        else:
            self.labels = None

    def __getitem__(self, index):
        str = random.choice(list(self.labels))
        img = f = self.files[index]

        if self.labels is not None:
            i = os.path.splitext(os.path.basename(f))[0]
            i = i[-6:]
            # print(i)
            if i in self.labels.keys():
                target = self.labels[i]
            else:
                str = random.choice(list(self.labels))
                f = glob.glob(os.path.join(self.dir, "*/model_" + str + ".obj"))
                img = f[0]
                target = self.labels[str]
            # print(target)
            if self.target_transform is not None:
                target = self.target_transform(target)

        if self.transform is not None:
            # Using the predefine methods.
            # eg. ToMesh()&ProjectOnSphere()
            img = self.transform(img)

            return img, target
        else:
            return img

    def __len__(self):
        return len(self.files)

    def _check_exists(self):
        PATH = os.path.join(self.dir, "*.obj")
        files = glob.glob(os.path.join(self.dir, "*/*.obj"))

        return len(files) > 0

    def _download(self, url):
        import requests

        filename = url.split('/')[-1]
        file_path = os.path.join(self.root, filename)

        if os.path.exists(file_path):
            return file_path

        print('Downloading ' + url)

        r = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()

        return file_path

    def _unzip(self, file_path):
        import zipfile

        if os.path.exists(self.dir):
            return

        print('Unzip ' + file_path)

        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()
        os.unlink(file_path)

    def _fix(self):
        print("Fix obj files")

        r = re.compile(r'f (\d+)[/\d]* (\d+)[/\d]* (\d+)[/\d]*')

        path = os.path.join(self.dir, "*.obj")
        files = sorted(glob.glob(path))

        c = 0
        for i, f in enumerate(files):
            with open(f, "rt") as x:
                y = x.read()
                yy = r.sub(r"f \1 \2 \3", y)
                if y != yy:
                    c += 1
                    with open(f, "wt") as x:
                        x.write(yy)
            print("{}/{}  {} fixed    ".format(i + 1, len(files), c), end="\r")

    def download(self, dataset, perturbed):

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = self.url_data.format(dataset + ("_perturbed" if perturbed else ""))
        file_path = self._download(url)
        self._unzip(file_path)
        self._fix()

        if dataset != "test":
            url = self.url_label.format(dataset)
            self._download(url)

        print('Done!')
