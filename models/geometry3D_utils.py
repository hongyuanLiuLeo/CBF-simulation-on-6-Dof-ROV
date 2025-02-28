from abc import ABCMeta,abstractmethod
import casadi as ca
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polytope as pt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

class ConvexRegion3D:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_convex_rep(self):
        raise NotImplementedError()

    @abstractmethod
    def get_plot_patch(self, ax):
        raise NotImplementedError()


# class BoxRegion(ConvexRegion3D):
#     """[3D Box shape]"""

#     def __init__(self, left, right, down, up, bottom, top):
#         self.left = left
#         self.right = right
#         self.down = down
#         self.up = up
#         self.bottom = bottom
#         self.top = top

#     # get A and b for box region
#     def get_convex_rep(self):
#         mat_A = np.array([
#             [-1, 0, 0], [0, -1, 0], [0, 0, -1],
#             [1, 0, 0], [0, 1, 0], [0, 0, 1]
#         ])
#         vec_b = np.array([
#             [-self.left], [-self.down], [-self.bottom],
#             [self.right], [self.up], [self.top]
#         ])
#         return mat_A, vec_b

#     def get_plot_patch(self, ax):
#         # Vertices of the box in 3D space
#         vertices = [
#             [self.left, self.down, self.bottom],
#             [self.right, self.down, self.bottom],
#             [self.right, self.up, self.bottom],
#             [self.left, self.up, self.bottom],
#             [self.left, self.down, self.top],
#             [self.right, self.down, self.top],
#             [self.right, self.up, self.top],
#             [self.left, self.up, self.top]
#         ]
        
#         # Faces of the box
#         faces = [
#             [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
#             [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
#             [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
#             [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
#             [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
#             [vertices[4], vertices[7], vertices[3], vertices[0]],  # left
#         ]
        
#         # Plot the 3D box
#         ax.add_collection3d(Poly3DCollection(faces, 
#                                              facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))


class PolytopeRegion3D(ConvexRegion3D):
    def __init__(self,mat_A,vec_b):
        self.mat_A = mat_A
        self.vec_b = vec_b
        self.points = pt.extreme(pt.Polytope(mat_A,vec_b))

    # generate A and b from convex points
    @classmethod
    def convex_hull(self,points):
        """Convex hull of N points in d dimensions as Nxd numpy array"""
        P = pt.reduce(pt.qhull(points))
        return PolytopeRegion3D(P.A,P.b)
    
    # get A and b
    def get_convex_rep(self):
        # TODO: Move this change into constructor instead of API here
        return self.mat_A, self.vec_b.reshape(self.vec_b.shape[0],-1)
    
    def get_plot_patch(self, ax):
        """
        Plot the polytope in 3D using Poly3DCollection.
        ax: 3D matplotlib axis
        """
        # 将 self.points 转换为 NumPy 数组
        points = np.array(self.points)

        # 使用 scipy.spatial.ConvexHull 生成面
        hull = ConvexHull(points)

        # 获取多面体的面 (hull.simplices 是点的索引)
        faces = [points[simplex] for simplex in hull.simplices]

        # 使用 Poly3DCollection 绘制 3D 面
        poly3d = Poly3DCollection(faces, alpha=0.3, edgecolor='r', facecolor='cyan', linewidths=1)

        # 将多面体添加到 3D 图中
        ax.add_collection3d(poly3d)



