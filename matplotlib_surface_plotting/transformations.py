"""3D transformation matrix utilities for surface plotting."""

import numpy as np
from typing import Tuple


def frustum(left: float, right: float, bottom: float, top: float, 
           znear: float, zfar: float) -> np.ndarray:
    """Create a frustum projection matrix.
    
    Args:
        left, right: Left and right clipping planes
        bottom, top: Bottom and top clipping planes  
        znear, zfar: Near and far clipping planes
        
    Returns:
        4x4 frustum projection matrix
    """
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[1, 2] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M


def perspective(fovy: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """Create a perspective projection matrix.
    
    Args:
        fovy: Field of view angle in degrees
        aspect: Aspect ratio (width/height)
        znear, zfar: Near and far clipping planes
        
    Returns:
        4x4 perspective projection matrix
    """
    h = np.tan(0.5 * np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def translate(x: float, y: float, z: float) -> np.ndarray:
    """Create a translation matrix.
    
    Args:
        x, y, z: Translation amounts along each axis
        
    Returns:
        4x4 translation matrix
    """
    return np.array([[1, 0, 0, x], [0, 1, 0, y],
                     [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


def xrotate(theta: float) -> np.ndarray:
    """Create a rotation matrix around the X axis.
    
    Args:
        theta: Rotation angle in degrees
        
    Returns:
        4x4 rotation matrix
    """
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0,  0, 0], [0, c, -s, 0],
                     [0, s,  c, 0], [0, 0,  0, 1]], dtype=float)


def yrotate(theta: float) -> np.ndarray:
    """Create a rotation matrix around the Y axis.
    
    Args:
        theta: Rotation angle in degrees
        
    Returns:
        4x4 rotation matrix
    """
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s, 0], [ 0, 1, 0, 0],
                     [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=float)


def zrotate(theta: float) -> np.ndarray:
    """Create a rotation matrix around the Z axis.
    
    Args:
        theta: Rotation angle in degrees
        
    Returns:
        4x4 rotation matrix
    """
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, -s, 0, 0], 
                     [ s,  c, 0, 0],
                     [ 0,  0, 1, 0], 
                     [ 0,  0, 0, 1]], dtype=float)


def create_mvp_matrix(view_angle: float, z_rotate: float, x_rotate: float, 
                     flat_map: bool = False) -> np.ndarray:
    """Create a complete Model-View-Projection matrix.
    
    Args:
        view_angle: Viewing angle for Y rotation
        z_rotate: Z-axis rotation angle
        x_rotate: X-axis rotation angle  
        flat_map: Whether this is for flat map rendering
        
    Returns:
        4x4 MVP transformation matrix
    """
    mvp = (perspective(25, 1, 1, 100) @ 
           translate(0, 0, -3) @ 
           yrotate(view_angle) @ 
           zrotate(z_rotate) @ 
           xrotate(x_rotate) @ 
           zrotate(270 * flat_map))
    return mvp