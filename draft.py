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

    theta, phi = S2.meshgrid(b=b, grid_type='SOFT')

    sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
    #sgrid = sgrid.reshape((-1, 3))



    #R = rotmat(alpha, beta, gamma, hom_coord=False)
    #sgrid = np.einsum('ij,nj->ni', R, sgrid)#rotation



    return sgrid


def s2_equatorial_grid(max_beta=0, n_alpha=32, n_beta=1):
    '''
    :return: rings around the equator
    size of the kernel = n_alpha * n_beta
    '''
    beta = np.linspace(start=np.pi/2 - max_beta, stop=np.pi/2 + max_beta, num=n_beta, endpoint=True)
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False)
    B, A = np.meshgrid(beta, alpha, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    grid = np.stack((B, A), axis=1)
    return tuple(tuple(ba) for ba in grid)

fig=plt.figure()
sgrid =make_sgrid(4,0,0,0)
grid=s2_equatorial_grid()
x=sgrid[:,0]
y=sgrid[:,1]
z=sgrid[:,2]
x=x.reshape(-1,1)
y=y.reshape(-1,1)
z=z.reshape(-1,1)
ax=Axes3D(fig)
ax.scatter(x,y,z)

#plt.legend()
plt.show()

def inverse_render_model(points:np.ndarray,sgrid:np.ndarray):
    #wait for implementing
    print("Aloha")
    #Fistly, we get the centroid, then translate the points
    x=np.sum(points[:,0])
    y=np.sum(points[:,1])
    z=np.sum(points[:,2])
    centroid=np.array([x,y,z])
    points-=centroid

    #After normalization, compute the distance between the sphere and points
    x=np.sum(points[:,0])
    y=np.sum(points[:,1])
    z=np.sum(points[:,2])
    radius=np.sqrt(x**2+y**2+z**2)
    dist=1-(1/np.max(radius))*radius
    dist_im=np.ones(sgrid.shape[0],sgrid.shape[1])#signal of distance from points to sphere



    #Projection
    from lie_learn.spaces import S2
    radius=np.repeat(radius,3).reshape(-1,3)
    points_on_sphere=points/radius
    theta,phi=S2.change_coordinates(points_on_sphere,p_from='C', p_to='S')



    #Interpolate
    b=np.sqrt(sgrid.shape[0])/2#bandwidth
    #By computing the m/n, we can find
    #the neighbours on the sphere
    m=phi/(np.pi/b)
    n=theta/(np.pi/(2*b))

    #need more validation
    nw=sgrid[n,m]
    ne=sgrid[n,m+1]
    sw=sgrid[n+1,m]
    se=sgrid[n+1,m+1]
    #=========================

    #calculate distance and relevant weight
    nw_dist=np.sqrt((nw-points_on_sphere)**2)
    ne_dist=np.sqrt((ne-points_on_sphere)**2)
    sw_dist=np.sqrt((sw-points_on_sphere)**2)
    se_dist=np.sqrt((se-points_on_sphere)**2)
    sum=nw_dist+ne_dist+sw_dist+se_dist
    nw_weight=nw_dist/sum
    ne_weight=ne_dist/sum
    sw_weight=sw_dist/sum
    se_weight=se_dist/sum
    #save the signal of distance
    dist_im[n,m]-=radius[:,0]*nw_weight
    dist_im[n,m+1]-=radius[:,0]*ne_weight
    dist_im[n+1,m]-=radius[:,0]*sw_weight
    dist_im[n+1,m+1]-=radius[:,0]*se_weight

    im=dist_im
    return im


class ProjectFromPointsOnSphere:
    #Wait for implementing
    def __init__(self,bandwidth):
        self.bandwidth = bandwidth
        self.sgrid = make_sgrid(bandwidth, alpha=0, beta=0, gamma=0)  # create a sphere grid
        print("Aloha")

    def __call__(self,points):
        im=inverse_render_model(points,self.sgrid)
        im=im.astype(np.float32)
        return im

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)