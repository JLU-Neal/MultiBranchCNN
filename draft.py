import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from mpl_toolkits.mplot3d import Axes3D


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


def make_sgrid(b, alpha, beta, gamma):
    from lie_learn.spaces import S2

    beta, alpha = S2.meshgrid(b=b, grid_type='SOFT')

    sgrid = S2.change_coordinates(np.c_[beta[..., None], alpha[..., None]], p_from='S', p_to='C')

    # sgrid = sgrid.reshape((-1, 3))

    # R = rotmat(alpha, beta, gamma, hom_coord=False)
    # sgrid = np.einsum('ij,nj->ni', R, sgrid)#rotation

    return sgrid


def s2_equatorial_grid(max_beta=0, n_alpha=32, n_beta=1):
    '''
    :return: rings around the equator
    size of the kernel = n_alpha * n_beta
    '''
    beta = np.linspace(start=np.pi / 2 - max_beta, stop=np.pi / 2 + max_beta, num=n_beta, endpoint=True)
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    B, A = np.meshgrid(beta, alpha, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    grid = np.stack((B, A), axis=1)
    return tuple(tuple(ba) for ba in grid)

def linear_regression(X:np.ndarray,Y:np.ndarray):
    from sklearn.linear_model import LinearRegression
    lr=LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=False)
    reg=lr.fit(X,Y)
    coef=reg.coef_
    intercept=reg.intercept_

def inverse_render_model(points: np.ndarray, sgrid: np.ndarray):
    # wait for implementing
    print("Aloha")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Fistly, we get the centroid, then translate the points
    centroid_x = np.sum(x) / points.shape[0]
    centroid_y = np.sum(y) / points.shape[0]
    centroid_z = np.sum(z) / points.shape[0]
    centroid = np.array([centroid_x, centroid_y, centroid_z])
    points=points.astype(np.float)
    points -= centroid

    # After normalization, compute the distance between the sphere and points

    radius = np.sqrt(points[:,0] ** 2 + points[:,1] ** 2 + points[:,2] ** 2)
    #dist = 1 - (1 / np.max(radius)) * radius


    # Projection
    from lie_learn.spaces import S2
    radius = np.repeat(radius, 3).reshape(-1, 3)
    points_on_sphere = points / radius
    #ssgrid = sgrid.reshape(-1, 3)
    # phi, theta = S2.change_coordinates(ssgrid, p_from='C', p_to='S')
    out=S2.change_coordinates(points_on_sphere,p_from='C', p_to='S')

    phi=out[...,0]
    theta=out[...,1]

    phi = phi % (np.pi)
    theta = theta % (np.pi * 2)

    # Interpolate
    b = sgrid.shape[0] / 2  # bandwidth
    # By computing the m/n, we can find
    # the neighbours on the sphere
    m = np.trunc((phi - np.pi / (4 * b)) / (np.pi / (2 * b)))
    m = m.astype(int)
    n = np.trunc(theta / (np.pi / b))
    n = n.astype(int)

    # use a mask to avoid the index is out of the boundary
    mask_m=m+1<sgrid.shape[0]
    mask_n=n+1<sgrid.shape[1]
    mask=mask_m*mask_n
    m=m[mask]
    n=n[mask]
    nw = sgrid[m, n]
    ne = sgrid[m, n+1]
    sw = sgrid[m+1, n]
    se = sgrid[m+1, n+1]
    # =========================

    # calculate distance and relevant weight
    nw_dist = np.sqrt(np.sum((nw - points_on_sphere[mask]) ** 2))
    ne_dist = np.sqrt(np.sum((ne - points_on_sphere[mask]) ** 2))
    sw_dist = np.sqrt(np.sum((sw - points_on_sphere[mask]) ** 2))
    se_dist = np.sqrt(np.sum((se - points_on_sphere[mask]) ** 2))
    sum = nw_dist + ne_dist + sw_dist + se_dist
    nw_weight = nw_dist / sum
    ne_weight = ne_dist / sum
    sw_weight = sw_dist / sum
    se_weight = se_dist / sum

    # save the signal of distance
    radius=radius[mask]
    dist_im = np.zeros(sgrid.shape[0:2])  # signal of distance from points to sphere
    weight_im=np.zeros(sgrid.shape[0:2])  # Since each grid point on the sphere could be affected by several different signals, we need to normalize the values.
    dist_im[m, n] += radius[:, 0] * nw_weight
    dist_im[m, n + 1] += radius[:, 0] * ne_weight
    dist_im[m + 1, n] += radius[:, 0] * sw_weight
    dist_im[m + 1, n + 1] += radius[:, 0] * se_weight
    weight_im[m, n] += nw_weight
    weight_im[m, n + 1] += ne_weight
    weight_im[m + 1, n] += sw_weight
    weight_im[m + 1, n + 1] += se_weight
    mask_weight=weight_im!=0
    dist_im[mask_weight]/=weight_im[mask_weight]
    dist_im=1-dist_im

    im = dist_im
    #=======================================================================================
    fig = plt.figure()
    grid = make_sgrid(32, 0, 0, 0)
    grid = grid.reshape((-1, 3))

    xx= grid[:, 0]
    yy = grid[:, 1]
    zz = grid[:, 2]
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    zz = zz.reshape(-1, 1)
    ax = Axes3D(fig)
    ax.scatter(0, 0, 0)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(points_on_sphere[:, 0], points_on_sphere[:, 1], points_on_sphere[:, 2])

    ax.scatter(nw[:, 0], nw[:, 1], nw[:, 2])
    ax.scatter(ne[:, 0], ne[:, 1], ne[:, 2])
    ax.scatter(sw[:, 0], sw[:, 1], sw[:, 2])
    ax.scatter(se[:, 0], se[:, 1], se[:, 2])
    ax.scatter(xx, yy, zz)
    # plt.legend()


    #draw line
    ax=fig.gca(projection='3d')
    zero=np.zeros(points_on_sphere.shape[0])
    ray_x=np.stack((zero,points_on_sphere[:,0]),axis=1).reshape(-1,2)
    ray_y=np.stack((zero,points_on_sphere[:,1]),axis=1).reshape(-1,2)
    ray_z=np.stack((zero,points_on_sphere[:,2]),axis=1).reshape(-1,2)
    for index in range(points_on_sphere.shape[0]):
        ax.plot(ray_x[index],ray_y[index],ray_z[index])
    plt.show()

    return im


class ProjectFromPointsOnSphere:
    # Wait for implementing
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.sgrid = make_sgrid(bandwidth, alpha=0, beta=0, gamma=0)  # create a sphere grid
        print("Aloha")

    def __call__(self, points):
        im = inverse_render_model(points, self.sgrid)
        im = im.astype(np.float32)
        return im

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)


pfpos = ProjectFromPointsOnSphere(32)
#a = np.array([[0, 0, -1],[0,0,1]]).reshape(-1, 3)
a=np.random.random((2500,3))
im = pfpos(a)
