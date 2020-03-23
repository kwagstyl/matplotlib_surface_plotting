import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M
def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)
def translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y],
                     [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)
def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0,  0, 0], [0, c, -s, 0],
                     [0, s,  c, 0], [0, 0,  0, 1]], dtype=float)
def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0], [ 0, 1, 0, 0],
                      [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=float)

def plot_surf(vertices, faces,overlay,rotate=270, cmap='viridis', filename='plot.png', label=False):
    """plot mesh surface with a given overlay
    vertices - vertex locations
    faces - triangles of vertex indices definings faces
    overlay - array to be plotted
    cmap - matplotlib colormap
    rotate - 270 for lateral on lh, 90 for medial
    """
    vertices=vertices.astype(np.float)
    F=faces.astype(int)
    vertices = (vertices-(vertices.max(0)+vertices.min(0))/2)/max(vertices.max(0)-vertices.min(0))
    if not isinstance(rotate,list):
        rotate=[rotate]
    
    #colours smoothed (mean) or median if label
    if label:
        colours = np.median(overlay[F],axis=1)
    else:
        colours = np.mean(overlay[F],axis=1)
    colours = (colours - colours.min())/(colours.max()-colours.min())
    C = plt.get_cmap(cmap)(colours)
     #make figure dependent on rotations
    fig = plt.figure(figsize=(6*len(rotate),6))
    for i,view in enumerate(rotate):
        MVP = perspective(25,1,1,100) @ translate(0,0,-3) @ yrotate(view) @ xrotate(270)
    #translate coordinates based on viewing position
        V = np.c_[vertices, np.ones(len(vertices))]  @ MVP.T
        V /= V[:,3].reshape(-1,1)
        V = V[F]
    #triangle coordinates
        T =  V[:,:,:2]
    #get Z values for ordering triangle plotting
        Z = -V[:,:,2].mean(axis=1)
    #sort the triangles based on their z coordinate. If front/back views then need to sort a different axis
        I = np.argsort(Z)
        T, s_C = T[I,:], C[I,:]
        ax = fig.add_subplot(1,len(rotate),i+1, xlim=[-1,+1], ylim=[-1,+1],aspect=1, frameon=False,
         xticks=[], yticks=[])
        collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=s_C, edgecolor=s_C)
        collection.set_alpha(1)
        ax.add_collection(collection)
        plt.subplots_adjust(left =0 , right =1, top=1, bottom=0,wspace=0, hspace=0)
    fig.savefig(filename,bbox_inches = 'tight',)
