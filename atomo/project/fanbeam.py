from __future__ import division
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

__author__ = 'thanasi'


Rsd = 1000
Rid = 400

ny,nx = 500, 500

class AngleError(Exception):
    pass


def get_rotation_mat_2d(phi):
    return np.array([[np.cos(phi), -np.sin(phi) ],
                     [np.sin(phi), np.cos(phi)]], dtype=np.float32)


def phi_to_u(phi):
    return Rsd * np.tan(phi + np.pi/2)

def source_ray(phi, theta, step):
    ray_length = Rsd / np.cos(phi)
    #nsteps = ray_length/step

    ## detector point
    ud = phi_to_u(phi)

    ## set up ray of proper length and the
    ray = np.arange(0, ray_length, step, dtype = np.float32)
    #ustep = step * np.sin(phi)
    x = np.ones_like(ray) * nx//2
    rayz = np.array(zip(x,ray), dtype=np.float32)
    rray = rotate_ray_2d(rayz, (nx//2,0), phi)

    if theta != 0:
        rray = rotate_ray_2d(rayz, (nx//2, Rid), theta)

    ## remember to switch to [y,x] image coordinates
    return rray[:,::-1]

def rotate_ray_2d(ray, origin, phi):

    o = np.array(origin, dtype=np.float32)[np.newaxis,:]

    ## adjust ray before rotation
    aray = ray - o
    ## generate and apply rotation matrix
    rot_mat = get_rotation_mat_2d(phi)
    rray = np.dot(rot_mat, aray.T).T

    ## readjust to reflect center coordinates
    return rray + o


def fan_beam_project(im):
    pass