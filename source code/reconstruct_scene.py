# from 2.6.4_lab.ipynb

# !pip install chainer
import subprocess
import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import cupy as np
import pickle

from matplotlib import pyplot as plt
import imageio
import math
from PIL import Image
import time

NORMAL_LOW = True
DISTANCE_LOW = True
HIT_DISTANCE = 15
TOTAL_NUM_PROBES = 1


TRACE_RESULT_MISS    = 0
TRACE_RESULT_HIT     = 1
TRACE_RESULT_UNKNOWN = 2
TEX_FACTOR = 2
SMALL_TEX_FACTOR = 2

TEX_SIZE_SMALL = 64.0 * SMALL_TEX_FACTOR;
INV_TEX_SIZE_SMALL = 1.0 / (64 * SMALL_TEX_FACTOR)

TEX_SIZE = 1024.0 * TEX_FACTOR
INV_TEX_SIZE = 1. / (1024.0 * TEX_FACTOR)
minThickness = 0.03;
maxThickness = 0.50;
num_channel = 3
# TODO: change scale factor
TRI_ENLARGE_FACTOR = 1
DIST_SCALE_FACTOR = 1 * TRI_ENLARGE_FACTOR;

# 9 columns: tmin(0), tmax(1), hit_probe_tex_coord x(2) y(3),
# relative probe index(4), probes_left(5), result(6)
# i nearestProbeIndices(7), hit_probe_idx(8)
# base_probe_idx (9)

# update tmin if unknown(one side hit, but not hit, i.e. backface) in high res trace
TMIN = num_channel;
# update tmax if hit in high res trace
TMAX = 1 + num_channel;
HIT_PROBE_TEX_X = 2 + num_channel;
HIT_PROBE_TEX_Y = 3 + num_channel;
# NEAREST_PROBE_IDX is the index of the probe in the cube cage that's closest to shading point
# RELATIVE_PROBE_IDX is NEAREST_PROBE_IDX's entry into teture2darray
RELATIVE_PROBE_IDX = 4 + num_channel;
PROBES_LEFT = 5 + num_channel; # onece hit, reset to 0
# 0 TRACE_RESULT_MISS, 1 TRACE_RESULT_HIT, 2 TRACE_RESULT_UNKNOWN
HIT_RESULT = 6 + num_channel;
NEAREST_PROBE_IDX = 7 + num_channel; # index of the probe in the cube cage that's closest to shading point

# helper function
def get_probe_info(PROBE_POS):
    # 8 probes
    probe_info = {'probeCounts': PROBE_COUNTS,
                  'probeStartPosition': PROBE_POS,
                  'probeStep': PROBE_STEP}
    return probe_info;

def normalize_bc(v, eps=1e-8):
    dist = np.linalg.norm(v, axis=1).reshape(v.shape[0], 1)
    return np.divide(v, (dist + eps))

def distance_bc(a, b):
    return np.linalg.norm(b - a, axis=1)

def sign_bc(v):
    neg_indices = np.where(v < 0)[0]
    pos_indices = np.where(v > 0)[0]

    sign = np.zeros((v.size, 1))
    sign[neg_indices] = -1.
    sign[pos_indices] = 1.
    return sign

def length_squared_bc(v):
    return np.linalg.norm(v, axis=1) ** 2; # 1d array

def sign_not_zero_bc(result_sign):
    result_sign[result_sign >= 0 ] = 1
    result_sign[result_sign < 0 ] = -1
    return result_sign

# 2d coordinates of the segment endpoints are the 2d projections of intersections of 3d line with
# 3 axis-aligned planes that pass throught the probe centers
def oct_encode_bc(v):
    """
    input: unit vector in probe space, i.e. cg.xyz
    output: octahedral vector on the [-1, 1] square, is cg.zx
    """
    v = np.roll(v, 1, axis=1)
    l1_norm_inv = np.divide(1., np.linalg.norm(v, ord=1, axis=1))
    result = v[:, :2] * l1_norm_inv.reshape(v.shape[0], 1)

    neg_indices = np.where(v[:, 2] < 0)[0]
    if neg_indices.size > 0:
        temp = 1 - abs(np.flip(result[neg_indices], axis=1))
        result[neg_indices] = temp * sign_not_zero_bc(np.copy(result[neg_indices]))
    return result;

def oct_decode_bc(o):
    """
    octahedral vector maps to 3d octahedron to sphere
    input
        octahedral vector on the [-1, +1] square, is cg.zx
    output
        output is converted to cg's axis by shifting 1st column to the last
        unit vector in probe space, i.e. cg.xyz
    """
    v = np.hstack((o, (1 - np.abs(o).sum(1)).reshape(-1, 1)))
    neg_indices = np.where(v[:, 2] < 0)[0]
    if neg_indices.size > 0:
          temp = 1- np.abs(np.flip(v[neg_indices][:, :2], axis=1))

          dot_prd = temp * sign_not_zero_bc(np.copy(v[neg_indices][:, :2]))
          v_neg = v[neg_indices]
          v_neg[:, :2] = dot_prd

          v[neg_indices] = v_neg
    v = normalize_bc(v)
    v = np.roll(v, -1, axis=1)
    return v

def uv_to_oct_vec(uv):
    return uv * 2 - 1

def oct_vec_to_uv(oct):
    return oct * 0.5 + 0.5

def get_params(cam, width, height):
    aspect_ratio = width / height
    world_to_camera = lookAt(cam.eye, cam.at, cam.up)
    camera_to_world = np.linalg.inv(world_to_camera)

    fov = cam.fov
    radians = fov * math.pi / 180
    h_c = 2 * np.tan(radians/2)
    w_c = aspect_ratio * h_c
    a, b = -w_c/2, h_c/2
    dx, dy = w_c/width, h_c/height

    return width, height, camera_to_world, dx, dy, a, b


# data structure
class Ray:
    def __init__(self, origin, dir, max=1e5):
        self.o = origin
        self.d = dir
        self.max = max

class Camera:
    def __init__(self, eye, at, up, fov=None):
        self.eye = eye
        self.at = at
        self.up = up
        self.fov = fov

class LightFieldSurface:
    def __init__(self,
                 radianceProbeGrid=[],
                 normalProbeGrid=[],
                 distanceProbeGrid=[],
                 lowResolutionDistanceProbeGrid=[],
                 probeCounts=[],
                 probeStartPosition=[],
                 probeStep=[],
                 lowResolutionDownsampleFactor=[],
                 lowResolutionNormalProbeGrid=[],
                 irradianceProbeGrid=[],
                 meanDistProbeGrid=[]):

        self.radianceProbeGrid = radianceProbeGrid;
        self.normalProbeGrid = normalProbeGrid;
        self.distanceProbeGrid = distanceProbeGrid;
        self.lowResolutionDistanceProbeGrid = lowResolutionDistanceProbeGrid;
        self.lowResolutionNormalProbeGrid = lowResolutionNormalProbeGrid;

        self.probeCounts = probeCounts;
        self.probeStartPosition = probeStartPosition;
        self.probeStep = probeStep;

        self.lowResolutionDownsampleFactor = lowResolutionDownsampleFactor;
        self.irradianceProbeGrid = irradianceProbeGrid;
        self.meanDistProbeGrid = meanDistProbeGrid;

def scanner_to_cg(eye):
    """transform from scanner's world space to cg's coordinate system"""
    x,y,z = eye[0], eye[1], eye[2]
    eye_cg = np.array([x, z, -y])
    return eye_cg


def lerp(x, v0, v1):
    """
    linear interpolation
    v0, v1: (#pixels, 3)
    x: distance from v0 to x， （#pixels, 1)
    """
    # print(f'v0 {v0.shape}')
    return v0 + x * (v1-v0)

def bilinear_interpolation(s, t, u00, u10, u01, u11, GAMMA=False):
    """
    https://sites.cs.ucsb.edu/~lingqi/teaching/resources/GAMES101_Lecture_09.pdf
    u00 - u11: (#pixels, 3), radiance at those pixels
    """
    # print(s, t)
    # print(f's \n {s}')
    u0 = lerp(s, u00, u10)
    u1 = lerp(s, u01, u11)
    val = lerp(t, u0, u1)
    return val;

def find_nearest_pixels(probe_uv_coords, size):
    """
    find nearest four pixels, bottom left, counterclockwise
    """
    u, v = probe_uv_coords[:, 0], probe_uv_coords[:, 1] # (2,)
    # print(f'u \n {u * size}')
    # print(f'v \n {v * size}')
    px = np.floor(u * size - 0.5)
    py = np.floor(v * size - 0.5)
    # print(f'px {repr(px)}')
    # print(f'py {repr(py)} \n')

    bottom_left = np.clip(np.vstack((py, px)).transpose(), 0, size-1)
    bottom_right = np.clip(np.vstack((py, px+1)).transpose(), 0, size-1)
    top_right = np.clip(np.vstack((py+1, px+1)).transpose(), 0, size-1)
    top_left = np.clip(np.vstack((py+1, px)).transpose(), 0, size-1)

    # print(f'bottom left \n {px}, \n {py}')
    # col, row
    nearest_neighbors = np.ones((probe_uv_coords.shape[0], 4, 2), dtype=np.uint16) # dtype
    nearest_neighbors[:, 0] = bottom_left
    nearest_neighbors[:, 1] = bottom_right
    nearest_neighbors[:, 2] = top_right
    nearest_neighbors[:, 3] = top_left

    # print(f'nearest_neighbors \n {nearest_neighbors}')
    return nearest_neighbors

def texel_fetch(uv_coord, name, probe_index, texture_maps=None):
    """
    uv_coord is texture coord of octahedral [0, 1]^2
    probe_index is relative probe index, i.e. entry of nearest probe into texture2darray
    probe_index: (num_pixels, )
    """
    num_pixels = probe_index.shape[0];
    expanded_tex_maps = None;

    size = 64 * SMALL_TEX_FACTOR if 'Low' in name else 1024 * TEX_FACTOR;
    idx_sort = np.argsort(probe_index)
    sorted_records_array = probe_index[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start.tolist()[1:])
    result = np.ones((num_pixels, 3)) * -1

    for i in range(vals.shape[0]):
        probe_idx = vals[i]
        c = count[i]
        # find uv coord of this probe_idx
        probe_uv_coords = uv_coord[idx_sort][idx_start[i]:idx_start[i]+c]

        if (name == 'radiance') and BILINEAR_INTERPOLATION:
            # print(f'=============== bilinear interpolation ===============')
            # (2,) u bottom up, v left to right, i.e. u is col, v is row
            u, v = probe_uv_coords[:, 0], probe_uv_coords[:, 1]
            y, x = u * size, v * size
            nearest_neighbors = find_nearest_pixels(probe_uv_coords, size)
            row = nearest_neighbors[:, :, 0]
            col = nearest_neighbors[:, :, 1]
            min_x = np.min(row, axis=1) + 0.5
            min_y = np.min(col, axis=1) + 0.5
            t = x - min_x
            s = y - min_y
            tm = texture_maps[probe_idx.get()]
            rad_neighbors = tm[size-1-row, col]
            val = bilinear_interpolation(s[:, None], t[:, None], rad_neighbors[:, 0], rad_neighbors[:, 1], \
                                  rad_neighbors[:, 3], rad_neighbors[:, 2],
                                  GAMMA=GAMMA) # u00, u10, u01, u11
        else:
            # print(f'====== name {name} ======')
            u, v = probe_uv_coords[:, 0], probe_uv_coords[:, 1] # (2,)
            col = (u * size).astype(int)
            row = (v * size).astype(int)
            col = np.clip(col, 0, size-1) # col, u: left to right
            row = np.clip(row, 0, size-1) # row, v: bottom up
            tm = texture_maps[probe_idx.get()]
            val = tm[size-1-row, col]
        result[res[i]] = val
    return result


def normalize(v, eps=1e-8):
    dist = np.linalg.norm(v)
    return v / (dist + eps)


def lookAt(eye, at, up):
    """
    Viewing transformation.
    Parameters:
        eye (np.array): eye postion
        at (np.array): the point the eye is looking at (usually, the center of
          an object of interest)
        up (np.array): up vector (vertically upward direction)
    """
    z = normalize(eye - at)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))
    A = np.column_stack((x, y, z, eye))
    A = np.row_stack((A, np.array([0, 0, 0, 1])))
    return np.linalg.inv(A)

def get_texture(name, size):
        im_path = f'{TEXTURE_MAPS}/{name}.png';
        image = Image.open(im_path)
        image_sequence = image.getdata()
        image_array = np.array(image_sequence)

        image_array = image_array.reshape((size, size, 3))
        image_array = image_array.astype(np.float32)/255.0
        return image_array

def show_and_save(img_buffer, path=None, save=True):
    img_buffer = img_buffer.reshape((-1, size, 3))
    # plt.imshow(img_buffer)
    # plt.show()

    if save:
        # timestr = datetime.now(EASTERN).strftime("%I%M%p_%S")
        # path = f'{DIRECTORY}/{NUM_PROBES}probes_{timestr}.png'
        img_buffer_img = img_buffer * 255
        img_uint8 = img_buffer_img.astype(np.uint8)
        img_buffer_img = Image.fromarray(img_uint8.get())
        # img_buffer_img = Image.fromarray(np.array(img_buffer_img).astype(np.uint8))
        img_buffer_img.save(path)
        # print(f'image saved successfully to \n {path}')

def create_pair(t1, t2, size=64):
    return np.asarray([t1[0] * size, t2[0] * size]), \
           np.asarray([size - 1 - t1[1] * size, size - 1 - t2[1] * size])




def getPixelCoordsAndRayDir(cam, width, height, jitter=False):
    """Show pixel coords in world space"""
    width, height, camera_to_world, dx, dy, a, b = get_params(cam, width, height);
    x = np.arange(width)
    y = np.arange(height)

    pixel_y_c = b - dy * (y + 0.5)
    pixel_x_c = a + dx * (x + 0.5)

    xx, yy = np.meshgrid(pixel_x_c, pixel_y_c)
    xx = xx.flatten()
    yy = yy.flatten()
    lst = np.stack((xx, yy, np.ones_like(xx) * -1, np.ones_like(yy)), axis=0)

    pixel_world = np.matmul(camera_to_world, lst)
    pixel_world = pixel_world / pixel_world[-1]
    ray_dirs = normalize_bc(np.array(pixel_world[:3].transpose()) - eye)
    return ray_dirs

def gridCoordToProbeIndex(probeCoords, L):
    return (probeCoords[:, 0] + probeCoords[:, 1] * L.probeCounts[0] + \
            probeCoords[:, 2] * L.probeCounts[0] * L.probeCounts[1]).astype(np.int16);


# TODO: return nearest probe indices and base probe index
# use grid iterator here
def nearest_probe_indices(origins, ray_dirs, L):
    """
    probe coords to entry
    origins is shading point in source code
    in an indexed cube, return index of the base probe closest to the shading point, usually 0
    base_probe_idx is the entry of probe0 in texture2darray
    i is index of the probe in the cube cage that's closest to shading point
    return nearest probe indices and base probe index
    """
    # for 16 probes, 3 cube cages, base probe index set it to 0, 1, 2
    num_pixel = ray_dirs.shape[0]
    width = np.sqrt(num_pixel).astype(np.uint16)
    l = np.floor(width/3).astype(np.int)
    base_index_matrix = np.empty((np.sqrt(num_pixel).astype(np.uint16), np.sqrt(num_pixel).astype(np.uint16)), dtype=np.uint16)

    base_index_matrix[:, 0:l] = 0
    base_index_matrix[:, l:l*2] = 1
    base_index_matrix[:, l*2:] = 2

    return np.array([0] * num_pixel, dtype=np.uint16), \
           base_index_matrix.reshape(num_pixel)

def relative_probe_index(base_probe_idx, relative_idx, L):
    """
    let's call probe0 as the probe in the indexed cube with index 0,
    i think:
    baseProbeIndex is the entry of probe0 in texture2darray
    relative_idx is index of the probe in the cube cage that's closest to shading point
    refer to fig3, indexed cube of probes
    return relative_idx's entry into teture2darray
    """
    num_probes = np.prod(L.probeCounts, axis=0)
    stride = np.array([1,
                       L.probeCounts[0].get(),
                       (L.probeCounts[0].get() * L.probeCounts[1].get())])
    size = relative_idx.shape[0]
    ones = np.array([1] * size)
    offset = np.vstack((np.bitwise_and(relative_idx, ones),
                      np.bitwise_and(np.right_shift(relative_idx, 1), ones),
                      np.bitwise_and(np.right_shift(relative_idx, 2), ones),
                      )).transpose()

    dot = np.tensordot(offset, stride, axes=([1], [0]))
    # why do we need bitwise_and?????????
    # probe_idx = np.bitwise_and(base_probe_idx + dot, num_probes - 1)
    return (base_probe_idx + dot) & (num_probes - 1);

def next_cycle_index(cycle_index):
    return np.bitwise_and(cycle_index + 3, 7)

# findmsb returns the bit number of the most
# significant bit in the binary representation of value.

# For positive integers, the result will be the bit number of the most significant bit set to 1.
# For negative integers, the result will be the bit number of the most significant bit set to 0.
# For a value of zero or negative one, -1 will be returned.
def findMSB(x):
    """x >= 0"""
    res = -1;
    for i in range(32):
        mask = np.right_shift(0x80000000, i)
        if np.bitwise_and(x, mask):
            res = 31 - i;
            break;
    return res

def grid_coord_to_position(c, L):
    """
    c: grid coord, probe xyz indices
    return c's position in world space
    """
    # print(f'c {c}')
    # print(f'L.probeStartPosition {L.probeStartPosition}')
    return np.asarray(L.probeStep) * c + np.asarray(L.probeStartPosition)

def probe_index_to_grid_coord(index, L):
    """
    Assumes probeCounts are powers of two.
    index: ProbeIndex, entry into L.radianceProbeGrid
    ProbeIndex: On [0, L.probeCounts.x * L.probeCounts.y * L.probeCounts.z - 1]
    return: grid coord of index
    """
    x = index & (L.probeCounts[0] - 1)
    y = np.right_shift(index & (L.probeCounts[0] * L.probeCounts[1]) - 1,
                       findMSB(L.probeCounts[0]))
    z = np.right_shift(index, findMSB(L.probeCounts[0] * L.probeCounts[1]))
    # print('probe_index_to_grid_coord \n', np.array([x, y, z]).transpose())
    return np.array([x, y, z]).transpose()

def probe_location(index, L):
    """
    ProbeIndex: index
    index is relative probe idx: relative_idx's entry in texture2darray
    (num_pixel, )
    """
    return grid_coord_to_position(probe_index_to_grid_coord(index, L), L)

def find_L(eye, p1, p2, L1, L2):
    """find the LightFieldSurface that's closes to the eye"""
    dist1 = np.linalg.norm(eye - p1)
    dist2 = np.linalg.norm(eye - p2)

    print(f'dist1 {dist1}')
    print(f'dist2 {dist2}')

    return (L1, L2) if dist1 < dist2 else (L2, L1)

def get_probe_position(oct_folder):
    with open(f'{oct_folder}/meta.txt') as f:
        lines = f.readlines()
        camera = np.array(lines[2].split(' ')[1:4], dtype=np.float32)
        return camera

def print_metadata(seconds):
    print('================= rendering results =================\n')
    # print(f'size {size}x{size}')
    print(f'camera at: {camera.eye}, fov {FOV}')
    print(f'looking at {camera.at} \n')

    print(f'GAMMA encode/decode: {GAMMA}')
    print(f'BILINEAR_INTERPOLATION: {BILINEAR_INTERPOLATION}')
    # print(f'probe_origin_degenerate: {probe_origin_degenerate}')
    # print(f'camera looking at {camera.at}')
    # print(f'number of probes {NUM_PROBES}')
    # print(f'Total number of probes {len(L.radianceProbeGrid)}')
    print(f'rendering time {round(seconds, 3)} s or {round(seconds/60, 3)} min')
    print('')


def computeRaySegments(origin, ray_dirs):
    """
    Segments a ray into the line segments that each lie within one Euclidean octant,
    which correspond to piecewise-linear projections in octahedral space.

    @param origin: vec3 probe_space_ray origin in probe space
    @param direction_frac: vec3 1 / probe_space_ray.direction
    @return boundaryT:  all boundary distance ("time") values in units of world-space distance
        along the ray. In the (common) case where not all five elements are needed, the unused
        values are all equal to tMax, creating degenerate ray segments.

    """
    # Time values for intersection with x = 0, y = 0, and z = 0 planes, sorted in increasing order
    t_min = ray_dirs[:, TMIN][:, None]
    t_max = ray_dirs[:, TMAX][:, None]

    boundaryTs = np.sort(np.multiply(np.divide(-1, ray_dirs[:, :3]),
                                  origin), -1)

    # Copy the values into the interval boundaries.
    # This expands at compile time and eliminates the
    # relative indexing, so it is just three conditional move operations
    boundaryTs = np.clip(boundaryTs, t_min, t_max)
    boundaryTs = np.c_[t_min, boundaryTs, t_max]

    return boundaryTs

# Returns the distance along v from the origin to the intersection
# with ray R (which it is assumed to intersect)

def dist_to_intersection_bc(ray, v, LOG=False):
    #  v.y * R.direction.z - v.z * R.direction.y;
    denom = v[:, 1] * ray.d[:, 2] - v[:, 2] * ray.d[:, 1]
    numer = np.ones(v.shape[0]) * np.inf

    threshold = 0.1 # TDOO 0.1, .001
    pos = np.abs(denom) > threshold
    pos_idx = np.where(np.abs(denom) > threshold)[0]
    neg_idx = np.where(np.abs(denom) <= threshold)[0]
    numer_pos = ray.o[pos_idx][:, 1] * ray.d[pos_idx][:, 2] - ray.o[pos_idx][:, 2] * ray.d[pos_idx][:, 1]
    numer_neg = ray.o[neg_idx][:, 0] * ray.d[neg_idx][:, 1] - ray.o[neg_idx][:, 1] * ray.d[neg_idx][:, 0]
    denom_neg = v[neg_idx][:, 0] * ray.d[neg_idx][:, 1] - v[neg_idx][:, 1] * ray.d[neg_idx][:, 0]
    numer[pos_idx] = numer_pos
    numer[neg_idx] = numer_neg
    denom[neg_idx] = denom_neg

    result = np.divide(numer, denom)
    result[np.isnan(result)] = 0

    return result;

def relativeProbeIndex(L, base_probe_idx, relative_idx):
    """
    relative_idx on [0, 7]
    TODO: Returns a probe index into L.radianceProbeGrid. It may be the *same* index as base_idx.
    """
    num_probes = np.prod(L.probe_counts, axis=0)
    stride = np.array([1, L.probe_counts[0], L.probe_counts[0] * L.probe_counts[1]])

    # refer to fig3, indexed cube of probes
    print(f'==== relative_idx {relative_idx} ====')
    offset = np.array([np.bitwise_and(relative_idx, 1),
                      np.bitwise_and(np.right_shift(relative_idx, 1), 1),
                      np.bitwise_and(np.right_shift(relative_idx, 2), 1),
                      ])
    dot = np.dot(offset, stride)
    # relative_idx's entry in texture2darray
    # why do we need this?????????
    probe_idx = (base_probe_idx + dot) & (num_probes - 1)
    return probe_idx

# same as 1.8.3 cpu probes
def low_resolution_trace_one_segment(probe_space_origin, probe_space_dirs,
                                     tex_coord, segment_end_tex_coord, # 0-1
                                     end_high_res_tex_coord, L):
    # print('\n ================================  START in LOW RES =============================== \n')

    epsilon = 0.001 #   // Ensure that we step just past the boundary, so that we're slightly inside the next texel, rather than at the boundary and randomly rounding one way or the other.
    num_pixels = probe_space_dirs.shape[0]
    P0 = tex_coord * TEX_SIZE_SMALL
    P1 = segment_end_tex_coord * TEX_SIZE_SMALL
    # P1[np.linalg.norm(P1 - P0, axis=1) ** 2  < 0.01] += 0.01
    P1[np.linalg.norm(P1 - P0, axis=1) ** 2  < 0.0001] += 0.01
    delta = P1 - P0 # (2, 2)

    permute = False;
    # print('permute \n', abs(delta[:, 0]) < abs(delta[:, 1]))

    # # Permute so that the primary iteration is in x to reduce large branches later
    # if (abs(delta[:, 0]) < abs(delta[:, 1])):
    #     permute = True;
    #     delta = np.flip(delta, axis=1)
    #     P0 = np.flip(P0, axis=1)
    #     P1 = np.flip(P1, axis=1)
    # print('permute ', permute)
    # print('P0', P0)
    # print('P1', P1)


    step_dir = sign_bc(delta[:, 0]) # input 1d array, output(2, 1)
    # ??why dp cant be (sign(1), sign(1)), dp is how much do we walk for each step
    dp = np.hstack((step_dir, delta[:, 1].reshape(-1, 1) * np.divide(step_dir, delta[:, 0].reshape(-1, 1)))) # inv_dx (2, 1), return (2, 2)
    # print('dp', dp) # why dp cant be (sign(1), sign(1))
    # print('delta', delta) # why dp cant be (sign(1), sign(1))

    # oct_coord = cg.zx, i.e. oct_coord[0] is z axis, oct_coord[1] is x axis
    initial_dir_from_probe = oct_decode_bc(uv_to_oct_vec(tex_coord)); # a unit vector input is an octahedral vector
    # TODO: if denom too small, it will be rounded to 0 so result will be nan
    # changed to compute distance between p to tex_coord
    prev_radial_dis_max_estimate = np.clip(dist_to_intersection_bc(Ray(probe_space_origin, probe_space_dirs[:, :3]), initial_dir_from_probe),
                                           0.0, np.inf)
    # prev_radial_dis_max_estimate = np.linalg.norm(P0, axis=1)
    # print(f'initial_dir_from_probe \n {initial_dir_from_probe}')

    abs_inv_py = 1. / abs(dp[:, 1]) # 1d array (num_pixels, )
    max_tex_coord_dist = length_squared_bc(segment_end_tex_coord - tex_coord) #(num_pixels, )

    count = 0
    hit = np.full(num_pixels, False)
    # hit, tex_coord_x, tex_coord_y, end_high_res_tex_coord x, end_high_res_tex_coord y
    # trace_result = np.array([[-1] * 5] * num_pixels).astype(np.float64)
    P = P0
    trace_result = np.array([[-1] * 5] * num_pixels).astype(np.float32)
    # print(f'create trace_result {trace_result.shape}')
    # while True:
    while count < 128:
        # print(f'=================== count {count} ===================')
        count += 1;
        # ================================= Part I =================================
        # find dist along each axis to edge of low res texel, so we can update endHighRes, P
        hit_idx = np.arange(num_pixels)[hit] # (0-16)

        # indices of pixeles that havent reached the end of seg and not hitted by the ray (4x4,)
        indices = np.where(P[:, 0].reshape(-1, 1) * sign_bc(delta[:, 0]) <=  P1[:, 0].reshape(-1, 1) * step_dir)[0]
        indices = indices[~np.in1d(indices, hit_idx)]
  
        if indices.size == 0: # all pixels are beyond the end of the seg
            # print('Low res - all pixels loop to end of segment or hit, break')
            not_hit = trace_result[:, 0] == -1
            trace_result[not_hit, 0] = TRACE_RESULT_MISS
            break;
        # TODO: permute is always false here
        probe_idx = probe_space_dirs[indices, RELATIVE_PROBE_IDX].astype(int) # (4x4,)

        # need to flip hit_pixel.y
        texture_name = 'distanceLow' if DISTANCE_LOW else 'distance'
        texture_maps = L.lowResolutionDistanceProbeGrid if DISTANCE_LOW else L.distanceProbeGrid
        scene_radial_dist_min_val = texel_fetch((np.flip(P[indices], axis=1) if permute else P[indices]) * INV_TEX_SIZE_SMALL,
                                                texture_name, probe_idx, texture_maps)[:, 0] * DIST_SCALE_FACTOR # (2, )
        # print(f'scene_radial_dist_min_val \n {scene_radial_dist_min_val}')
        # (4x4, 2) for sign_delta, intersection_pixel_dist
        sign_delta = np.hstack((sign_bc(delta[indices][:, 0]), sign_bc(delta[indices][:, 1])))
        intersection_pixel_dist = (sign_delta * 0.5 + 0.5) - sign_delta * np.modf(P[indices])[0]

        # take min otherwise the updated end_high_res_tex_coord would pass the current texel's boundary
        # how many dp.x, dp.y do we walk, i.e. how many steps do we walk for each axis (4x4, )
        ray_dist_to_next_pixel_edge = np.fmin(intersection_pixel_dist[:, 0], intersection_pixel_dist[:, 1] * abs_inv_py[indices]); # 1d (num_pixels, )

        # # ================================= Part II Scene Geometry Test =================================
        # # visible: if the low res texel part of scene geometry

        visible = indices
        visible_cond = np.full(indices.shape, True)

        P[indices[~visible_cond]] += dp[indices[~visible_cond]] * (ray_dist_to_next_pixel_edge[~visible_cond] + epsilon).reshape(-1, 1)

        if visible.size == 0:
            # Ensure that we step just past the boundary, so that we're slightly inside the next texel
            print('visible size == 0')
            continue;
        end_high_res_tex_coord[visible] = (P[visible] + dp[visible] * ray_dist_to_next_pixel_edge[visible_cond].reshape(-1, 1)) * INV_TEX_SIZE_SMALL
        end_high_res_tex_coord[visible] = np.flip(end_high_res_tex_coord[visible], axis=1) if permute else end_high_res_tex_coord[visible];
        beyond_seg = visible[length_squared_bc(end_high_res_tex_coord[visible] - tex_coord[visible]) > max_tex_coord_dist[visible]] # [T, T]
        end_high_res_tex_coord[beyond_seg] = segment_end_tex_coord[beyond_seg]

        # (<=4x4, 3), (<=4x4, )
        dir_from_probe = oct_decode_bc(uv_to_oct_vec(end_high_res_tex_coord[visible]));

        # changed when eye aligns with probe
        dist_from_probe_to_ray = np.clip(dist_to_intersection_bc(Ray(probe_space_origin[visible], \
                                  probe_space_dirs[visible, :3]), dir_from_probe, LOG=True),
                                         0.0, np.inf) # distance to end of current texel, (num_Pixels, )
        # dist_from_probe_to_ray = np.linalg.norm(end_high_res_tex_coord[visible] * TEX_SIZE_SMALL)

        max_radial_ray_dist = np.fmax(dist_from_probe_to_ray, prev_radial_dis_max_estimate[visible])
        prev_radial_dis_max_estimate[visible] = dist_from_probe_to_ray
        # ================================= Part III One side hit test =================================
        # if the texel is hitted, update traceResult, texCoord
        # if the texel is NOT hitted, update P, no need to update texCoord
        # note that endHigh is already updated for both hitted and not hitted texels
        hitted_ray = visible[scene_radial_dist_min_val[visible_cond] < max_radial_ray_dist] # (<=4x4, )
        not_hit = visible[scene_radial_dist_min_val[visible_cond] >= max_radial_ray_dist] # (<=4x4, )
        # print(f'scene_radial_dist_min_val \n {scene_radial_dist_min_val[visible_cond]}')

        hit[hitted_ray] = True
        # print(f'hitted_ray \n {hitted_ray} \n')
        if hitted_ray.size > 0:
            trace_result[hitted_ray, 0] = TRACE_RESULT_HIT
            trace_result[hitted_ray, 1:3] = INV_TEX_SIZE_SMALL * (np.flip(P[hitted_ray], axis=1) if permute else P[hitted_ray]);
            trace_result[hitted_ray, 3:5] = end_high_res_tex_coord[hitted_ray]

        # print(f'=========== end of TRAVERSE in LOW RES {max_cpu} ===========')
        P[not_hit] += dp[not_hit] * (ray_dist_to_next_pixel_edge[visible_cond][scene_radial_dist_min_val[visible_cond] >= max_radial_ray_dist] + epsilon).reshape(-1, 1)
    not_hit = trace_result[:, 0] == -1
    trace_result[not_hit, 0] = TRACE_RESULT_MISS
    trace_result[not_hit, 1:3] = segment_end_tex_coord[not_hit]
    trace_result[not_hit, 3:5] = end_high_res_tex_coord[not_hit]
    # print(f'lowres trace_result {trace_result.shape[0]}')
    return trace_result

def get_surface_thickness_bc(origins, dirs, dist_along_ray, normal, dir_from_probe):
    surface_thickness = minThickness + ((maxThickness - minThickness) *
                        np.clip(np.sum(dirs * dir_from_probe, axis=1), 0., np.inf) * # 1d arr
                        (2.0 - abs(np.sum((dirs * normal), axis=1))) *
                        np.clip(dist_along_ray * 0.1, 0.05, 1.0));
    return surface_thickness # 1d arr


# end_tex_coord <--> end_high_res_tex_coord
# trace 1 low res texel
# low_res_result [hit, tex_coord_x, tex_coord_y, end_tex_coord x, end_tex_coord y]
def high_resolution_trace_one_ray_segment(probe_space_origin, probe_space_dirs,
                                          low_res_result, probe_idx, pixel_idx, L):
    # print('\n ==============  START in HIGH RES ==================')
    num_pixels = probe_space_dirs.shape[0]
    tex_coord_delta = low_res_result[:, 3:] - low_res_result[:, 1:3]; # (2, 2)
    tex_coord_dist = np.linalg.norm(tex_coord_delta, axis=1); # float, 1d arr
    tex_coord_dir = tex_coord_delta * np.divide(1.0, tex_coord_dist).reshape(-1, 1) # (2, 2)
    tex_coord_step = INV_TEX_SIZE * (tex_coord_dist / np.amax(abs(tex_coord_delta), axis=1)) # 1d arr
    # print(f'probe_space_dirs {probe_space_dirs.shape}')

    dir_from_probe_before = oct_decode_bc(uv_to_oct_vec(low_res_result[:, 1:3])) # update in the while loop, (2, 3)
    dist_from_probe_to_ray_before = np.clip(dist_to_intersection_bc(Ray(probe_space_origin, probe_space_dirs[:, :3]), dir_from_probe_before),
                                            0.0, np.inf)
    # dist_from_probe_to_ray_before = np.linalg.norm(low_res_result[:, 1:3], axis=1)

    high_res_result = np.array([[np.inf]] * num_pixels)
    # print('tex_coord_dist', tex_coord_dist)

    # start of while loop
    # d = np.zeros(num_pixels).astype(np.float64) # 1d arr
    d = np.zeros(num_pixels).astype(np.float32) # 1d arr
    count = 0;

    # PRINT = False

    while True:
        # if count > 30:
        # print(f'========== TRAVERSE in HIGxH RES {count} ===========')
        count += 1;
        # 1st indices [not_end] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # pixeles that havent reached the end of the segment, and not hit or unknown
        indices = np.where(d <= tex_coord_dist)[0]
        indices = indices[~np.in1d(indices,
                                   np.where(high_res_result[:, 0] != np.inf)[0])]
        if indices.size == 0:
            # print(f'high_res_result {high_res_result}')
            # print('all pixeles reach the end of segment or unknown, return')
            break;
        tex_coord = tex_coord_dir[indices] * np.fmin(d[indices] + tex_coord_step[indices] * 0.5, tex_coord_dist[indices]).reshape(-1, 1) + low_res_result[:, 1:3][indices]; # half of 1 unit in uv coord
        probe_idx = probe_space_dirs[indices, RELATIVE_PROBE_IDX].astype(int)

        dist_from_probe_to_surface = texel_fetch(tex_coord, 'distance', probe_idx, L.distanceProbeGrid)[:, 0] * DIST_SCALE_FACTOR # 1d arr
        dir_from_probe = oct_decode_bc(uv_to_oct_vec(tex_coord)); # (2, 3)

        # (2, 2)
        tex_coord_after = tex_coord_dir[indices] * np.fmin(d[indices] + tex_coord_step[indices], tex_coord_dist[indices]).reshape(-1, 1) + low_res_result[:, 1:3][indices];
        dir_from_probe_after = oct_decode_bc(uv_to_oct_vec(tex_coord_after)); # (2, 3)
        # 1d arr
        dist_from_probe_to_ray_after = np.clip(dist_to_intersection_bc(Ray(probe_space_origin[indices], probe_space_dirs[indices][:, :3]), \
                                                                       dir_from_probe_after, LOG=True), 0.0, np.inf)
        # dist_from_probe_to_ray_after = np.linalg.norm(tex_coord_after * TEX_SIZE, axis=1)
        max_dist_from_probe_to_ray = np.fmax(dist_from_probe_to_ray_before[indices], dist_from_probe_to_ray_after) # 1d arr

        # print(f'dist_to_intersection_bc {dist_to_intersection_bc(Ray(probe_space_origin[indices], probe_space_dirs[indices][:, :3]), dir_from_probe_after)}')

        # 2.1 indices [one_side_hit] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # one_side_hit = max_dist_from_probe_to_ray >= dist_from_probe_to_surface

        # one_side_hit = np.logical_and(max_dist_from_probe_to_ray >= dist_from_probe_to_surface, \
        #                               dist_from_probe_to_surface != 0)
        one_side_hit = np.logical_and(max_dist_from_probe_to_ray >= dist_from_probe_to_surface, \
                                      dist_from_probe_to_surface > distance_degeneration)
        
        # one_side_hit_trial = indices[max_dist_from_probe_to_ray >= dist_from_probe_to_surface]
        one_side_hit_trial = indices[one_side_hit]
        # print(f'~~~~~~~~~~~~~~~~~~~~ 2.1 indices [one_side_hit] ~~~~~~~~~~~~~~~~~~~~ \n {one_side_hit} \n')

        # pixeles that have at least one side hit
        if one_side_hit.sum() == 0:
            dist_from_probe_to_ray_before[indices] = dist_from_probe_to_ray_after
            d[indices] += tex_coord_step[indices]
            # print('no pixeles have 1 side hit, continue')
            continue;

        # At least a one-sided hit; see if the ray actually passed through the surface, or was behind it
        dist_from_probe_to_ray_after_hit = dist_from_probe_to_ray_after[one_side_hit]

        min_dist_from_probe_to_ray = np.fmin(dist_from_probe_to_ray_before[one_side_hit_trial], dist_from_probe_to_ray_after_hit) # 1d arr
        dist_from_probe_to_ray = (max_dist_from_probe_to_ray[one_side_hit] + min_dist_from_probe_to_ray) * 0.5; # 1d arr
        probe_space_hit_point = dist_from_probe_to_ray.reshape(-1, 1) * dir_from_probe[one_side_hit] # (2, 3)
        # TODO: uncomment line 68
        # print('dist_from_probe_to_ray', dist_from_probe_to_ray.shape)
        # print('dist_from_probe_to_surface', dist_from_probe_to_surface[one_side_hit].shape)
        probe_space_hit_point = dist_from_probe_to_surface[one_side_hit].reshape(-1, 1) * dir_from_probe[one_side_hit] # (2, 3)
        dist_along_ray = np.sum((probe_space_hit_point - probe_space_origin[one_side_hit_trial]) \
                                  * probe_space_dirs[one_side_hit_trial][:, :3], axis=1); # 1d arr


        normal = texel_fetch(tex_coord[one_side_hit], 'normals', probe_idx[one_side_hit], L.normalProbeGrid)

        # print(f'normal \n {normal}')
        # print('tex_coord ', tex_coord[one_side_hit])
        no_one_side_hit_trial = indices[~np.in1d(indices, one_side_hit_trial)]
        # print('one_side_hit_trial', one_side_hit_trial)

        # TODO: verify d update
        d[indices] += tex_coord_step[indices]
        # d[no_one_side_hit_trial] += tex_coord_step[no_one_side_hit_trial]
        dist_from_probe_to_ray_before[no_one_side_hit_trial] = dist_from_probe_to_ray_after[~one_side_hit]
        visible_trial = one_side_hit_trial
        visible = np.full(visible_trial.shape, True)
        # print('normal', normal)
        # ==================== end of scene geo test ====================
        surface_thickness = get_surface_thickness_bc(probe_space_origin[visible_trial], probe_space_dirs[visible_trial][:, :3],
                                                     dist_along_ray[visible], normal[visible], dir_from_probe[one_side_hit][visible])

        # 4.1 indices [two_side_hit] 
        # Todo: add back surface thickness
        cond1 = (min_dist_from_probe_to_ray[visible] < (surface_thickness + dist_from_probe_to_surface[one_side_hit][visible]))
        cond2 = (np.sum(normal[visible] * probe_space_dirs[visible_trial][:, :3], axis=1) < 0.0)
        # print(f'cond1 {cond1} cond2 {cond2}')
        # print('surface_thickness', surface_thickness, '\n')
        two_side_hit = np.where(cond1 & cond2)[0]
        two_side_hit_trial = visible_trial[cond1 & cond2]
        ## TODO: comment below
        # two_side_hit, two_side_hit_trial = np.array([0]), np.array([0])

        # print(f'\n ~~~~~~~~~~~~~~~~~~ 4.1 indices one_side_hit visible [two_side_hit] {two_side_hit} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # display('========   two-sided hit =========', tex_coord_hit[visible][two_side_hit])
        probe_space_dirs[two_side_hit_trial, TMAX] = dist_along_ray[visible][two_side_hit]
        probe_space_dirs[two_side_hit_trial, HIT_PROBE_TEX_X:HIT_PROBE_TEX_Y+1] = tex_coord[one_side_hit][visible][two_side_hit]
        probe_space_dirs[two_side_hit_trial, HIT_RESULT] = TRACE_RESULT_HIT
        probe_space_dirs[visible_trial] = probe_space_dirs[visible_trial];
        probe_space_dirs[one_side_hit_trial] = probe_space_dirs[one_side_hit_trial];

        high_res_result[two_side_hit_trial, 0] = TRACE_RESULT_HIT
        # print(f'\n ~~~~~~~~~~~~~~~~~~ end 4.1 indices one_side_hit visible [two_side_hit]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')

        # 4.2 indices [no_hit] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        no_hit = np.where(~cond1 | ~cond2)[0]
        no_hit_trial = visible_trial[~cond1 | ~cond2]

        probe_space_hit_point_before = dist_from_probe_to_ray_before[no_hit_trial].reshape(-1, 1) * \
                                       dir_from_probe_before[indices][one_side_hit][visible][no_hit];
        dist_along_ray_before = np.sum((probe_space_hit_point_before - probe_space_origin[no_hit_trial]) *
                                        probe_space_dirs[visible_trial][no_hit, :3], axis=1); # 1d arr
        # 1d arr
        # t_min = np.fmax(visible_ray_dirs[no_hit, TMIN],
        #                 np.fmin(dist_along_ray[visible][no_hit], dist_along_ray_before))
        # print('distAlongRay', visible_ray_dirs[no_hit, TMIN])
        probe_space_dirs[no_hit_trial, TMIN] = np.fmax(probe_space_dirs[no_hit_trial, TMIN],
                                                  np.fmin(dist_along_ray[visible][no_hit], dist_along_ray_before))

        high_res_result[no_hit_trial, 0] = TRACE_RESULT_UNKNOWN
    # end of while loop
    high_res_result[high_res_result[:, 0] == np.inf] = TRACE_RESULT_MISS
    return high_res_result, probe_space_dirs

def trace_one_ray_segment(probe_space_origin, ray_dirs, segment_to_trace, L):
    # rayBumpEpsilon    = 0.0001; # meters TODO: what is this for, TUNE for DIFFERENT SCENE
    rayBumpEpsilon    = 0.001; # meters TODO: what is this for, TUNE for DIFFERENT SCENE
    num_pixels = ray_dirs.shape[0]
    probe_space_start = probe_space_origin + ray_dirs[:, :3] * (segment_to_trace[:, 0] + rayBumpEpsilon).reshape(num_pixels, 1)
    probe_space_end = probe_space_origin + ray_dirs[:, :3] * (segment_to_trace[:, 1] - rayBumpEpsilon).reshape(num_pixels, 1)
    # TODO: tex_coord.xy is cg.zx
    # tex_coord = oct_coord_to_uv_bc(start_oct_coord_bc) # uv [0, 1] square
    tex_coord = oct_vec_to_uv(oct_encode_bc(normalize_bc(probe_space_start))) # uv [0, 1] square
    segment_end_tex_coord = oct_vec_to_uv(oct_encode_bc(normalize_bc(probe_space_end)))
    result = np.array([[np.inf]] * num_pixels) # segment tracing result
    count = 0
    while True:
        end_tex_coord = np.ones((num_pixels, 2)) * np.inf;
        # print(f'\n ======= while loop trace_one_ray_segment {count} ========')
        count += 1;
        trace_dirs_trial = np.where(result[:, 0] == np.inf)[0]
        # print(f'result == np.inf {trace_dirs_trial.size}')
        if trace_dirs_trial.size == 0:
            # print('all pixeles are traced, BREAK')
            break;
        low_res_result = low_resolution_trace_one_segment(
                                probe_space_origin[trace_dirs_trial], ray_dirs[trace_dirs_trial],
                                tex_coord[trace_dirs_trial], segment_end_tex_coord[trace_dirs_trial],
                                end_tex_coord[trace_dirs_trial], L)

        # print(f'======== low_resolution_trace: hit, tex_coord, end_tex_coord ======== \n  {low_res_result}')

        low_miss_trial = trace_dirs_trial[low_res_result[:, 0] == TRACE_RESULT_MISS]
        result[low_miss_trial, 0] = TRACE_RESULT_MISS
        tex_coord[trace_dirs_trial] = low_res_result[:, 1:3]
        end_tex_coord[trace_dirs_trial] = low_res_result[:, 3:5]

        # ============================ LOW RES TRACE ============================
        low_hit = low_res_result[:, 0] == TRACE_RESULT_HIT # THIS IS CONSERVATIVE HIT NOT TWO SIDE HIT
        low_hit_trial = trace_dirs_trial[low_res_result[:, 0] == TRACE_RESULT_HIT] # THIS IS CONSERVATIVE HIT NOT TWO SIDE HIT

        if low_hit_trial.size == 0:
            # print('no pixel has 1 side hit in low res trace, break')
            break;
        ray_dirs_low_hit = ray_dirs[low_hit_trial]
        tex_coord_low_hit = tex_coord[low_hit_trial]
        # if high res trace hit or unknown, mark result, ray_dirs, if high res trace miss, loop again

        # ============================ HIGH RES TRACE ============================
        high_res_result, ray_dirs_high_res = high_resolution_trace_one_ray_segment(
                                                            probe_space_origin[low_hit_trial], ray_dirs_low_hit,
                                                            low_res_result[low_hit], 0, 0, L);

        # update ray_dirs and result: hit or unkown, hit probe idx
        # if not missed, it means high-resolution hit or went behind something,
        # which must be the result for the whole segment trace
        high_res_not_missed = np.where(high_res_result[:, 0] != TRACE_RESULT_MISS)[0]
        high_res_not_missed_trial = low_hit_trial[high_res_result[:, 0] != TRACE_RESULT_MISS]

        ray_dirs_low_hit[high_res_not_missed, HIT_RESULT] = high_res_result[high_res_not_missed, 0];
        ray_dirs[low_hit_trial] = ray_dirs_low_hit
        result[high_res_not_missed_trial, 0] = high_res_result[high_res_not_missed, 0]

        # If high resolution trace reached the end of the segment, and we've failed to find a hit, then mark result as miss, don't trace
        tex_coord_ray_direction_low_hit = normalize_bc(segment_end_tex_coord[low_hit_trial] - tex_coord_low_hit);
        # pixels that are traced to the end of the segment in high res trace
        end_seg_cond = np.sum(tex_coord_ray_direction_low_hit
                                  * (segment_end_tex_coord[low_hit_trial]
                                     - end_tex_coord[low_hit_trial]), axis=1) <= INV_TEX_SIZE / 2 # TODO: change to INV_TEX_SIZE
                                    #  - end_tex_coord[low_hit_trial]), axis=1) <= INV_TEX_SIZE
        end_seg = np.where(end_seg_cond)[0]
        end_seg = end_seg[~np.in1d(end_seg, high_res_not_missed)]
        end_seg_trial = low_hit_trial[end_seg]

        # result_low_hit[end_seg, 0] = TRACE_RESULT_MISS;
        result[end_seg_trial, 0] = TRACE_RESULT_MISS;
        tex_coord_low_hit[~end_seg_cond] = end_tex_coord[low_hit_trial][~end_seg_cond] \
                                      + tex_coord_ray_direction_low_hit[~end_seg_cond] * INV_TEX_SIZE * 0.1
        tex_coord[low_hit_trial] = tex_coord_low_hit

    # print(f'======== while loop in traceOneRaySeg {count} ========')
    return ray_dirs;


def trace_one_probe_oct(ray_origin, ray_dirs, L):
    print(f'enter trace one probe')
    num_pixels = ray_dirs.shape[0] # e.g. 2, 12

    degenerate_epsilon = 0.001; # meters, try 0; How short of a ray segment is not worth tracing?
    # how far should probe origin shift to avoid salt and pepper aliasing
    probe_origin = probe_location(ray_dirs[:, RELATIVE_PROBE_IDX].astype(int), L)
    probe_space_origin = ray_origin - probe_origin
    boundaryTs = computeRaySegments(probe_space_origin, ray_dirs); # ([1, 4, 6, 5])
    segments = np.zeros(num_pixels).astype(int) # i, only valid for 0-3

    # if eye and probe aligns, skip hierarchical tracing
    if np.all(probe_space_origin == 0):
        print(f'Eye and probe aligns \n')

        rayBumpEpsilon    = 0.001; # meters TODO: what is this for, TUNE for DIFFERENT SCENE
        segment_to_trace = np.zeros((ray_dirs.shape[0], 2))
        segment_to_trace[:, 1] = HIT_DISTANCE

        probe_space_start = probe_space_origin + ray_dirs[:, :3] * (segment_to_trace[:, 0] + rayBumpEpsilon).reshape(num_pixels, 1)
        probe_space_end = probe_space_origin + ray_dirs[:, :3] * (segment_to_trace[:, 1] - rayBumpEpsilon).reshape(num_pixels, 1)

        tex_coord = oct_vec_to_uv(oct_encode_bc(normalize_bc(probe_space_start))) # uv [0, 1] square
        # segment_end_tex_coord = oct_vec_to_uv(oct_encode_bc(normalize_bc(probe_space_end)))
        ray_dirs[:, HIT_RESULT] = TRACE_RESULT_HIT
        ray_dirs[:, HIT_PROBE_TEX_X:HIT_PROBE_TEX_Y+1] = tex_coord
        # print(f'tex_coord \n {tex_coord}')
        # print(f'ray_dirs \n {ray_dirs[:, HIT_PROBE_TEX_X:HIT_PROBE_TEX_Y+1]}')
        return ray_dirs

    # if eye and probe aligns, do hierarchical tracing
    probe_space_origin += probe_origin_degenerate
    count = 0;
    while True:
        print(f'=============== trace_one_ray_segment {count} ===============')
        count += 1
        # going to trace one segment
        # only trace the missed, or pixeles that are not traced

        ray_dirs[:, HIT_RESULT] = np.where(ray_dirs[:, HIT_RESULT] == TRACE_RESULT_UNKNOWN,
                                           -1, ray_dirs[:, HIT_RESULT])

        segments_indices = np.where(segments < 4)[0]
        segments_indices = segments_indices[~np.in1d(segments_indices,
                                                     np.where((ray_dirs[:, HIT_RESULT] == TRACE_RESULT_HIT)
                                                              | (ray_dirs[:, HIT_RESULT] == TRACE_RESULT_UNKNOWN))[0])]


        if segments_indices.size == 0:
            # print('no segment to trace')
            # print('===== finishing tracing 4 segments =====')
            # print('not_hit_or_unknown \n', np.where(not_hit_or_unknown)[0], '\n')
            ray_dirs[(ray_dirs[:, HIT_RESULT] != TRACE_RESULT_UNKNOWN) & (ray_dirs[:, HIT_RESULT] != TRACE_RESULT_HIT)
                      , HIT_RESULT] = TRACE_RESULT_MISS;
            return ray_dirs;

        i = segments[segments_indices]
        segment_to_trace = np.vstack((boundaryTs[segments_indices, i], boundaryTs[segments_indices, i+1])).transpose() # e.g. [[0. 3.19381204], [0., 3.10916528]]
        trace_bool = segments_indices[(segment_to_trace[:, 1] - segment_to_trace[:, 0]) > degenerate_epsilon]
        # print('segment_to_trace \n', segment_to_trace)
        # print(f'==================== new segment {segments[0]} ======================')

        if trace_bool.size > 0:
            segment_to_trace = segment_to_trace[(segment_to_trace[:, 1] - segment_to_trace[:, 0]) > degenerate_epsilon]

            # ================================ trace_ one_ray_segment =================================================
            # print(f'\n============ going to trace_one_ray_segment {count-1} ============')
            seg_time = time.time()

            ray_dirs[trace_bool] = trace_one_ray_segment(probe_space_origin[trace_bool],
                                                              ray_dirs[trace_bool],
                                                              segment_to_trace, L);
            temp_time = time.time()
            print(f'time for trace_one_ray_segment:', temp_time - seg_time, " s")
            # print(f'ray_dirs \n {ray_dirs[trace_bool, HIT_RESULT]}')
            # print(f'ray_dirs \n {ray_dirs[trace_bool, HIT_PROBE_TEX_X:HIT_PROBE_TEX_Y+1]}')
        segments[segments_indices] += 1


def find_next_L(eye, L1, L2, i, L3=None, L4=None):
    """
    find the LightFieldSurface that's ith closest to the eye
    i starts from 0
    """
    dist1 = np.linalg.norm(eye - L1.probeStartPosition).item()
    dist2 = np.linalg.norm(eye - L2.probeStartPosition).item()
    if L3:
        dist3 = np.linalg.norm(eye - L3.probeStartPosition).item()
        dist4 = np.linalg.norm(eye - L4.probeStartPosition).item() if L4 else np.inf
        dmap = {dist1: L1, dist2: L2, dist3: L3, dist4: L4}
        d = sorted([dist1, dist2, dist3, dist4])[i]
        # print(f'dist1 {dist1}')
    else:
        dmap = {dist1: L1, dist2: L2}
        d = sorted([dist1, dist2])[i]
    return dmap[d]

def trace(origin, ray_dirs, cell_index, L_LAB1, L_LAB2, L_LAB3=None, L_LAB4=None):
    """
    Traces a ray against the full lightfield.
    Returns true on a hit and updates tMax if there is a ray hit before tMax.
    Otherwise returns false and leaves tMax unmodified
    origin: ray origin
    ray_dirs (num_pixels, num_channel), i.e. flattened
    """
    # let's call probe0 as the probe in the indexed cube with index 0, then
    # base_probe_idx is the entry of probe0 in texture2darray

    # i, base_probe_idx = nearest_probe_indices(origin, ray_dirs, L)
    # i is index of the probe in the cube cage that's closest to shading point
    base_probe_idx = np.zeros(ray_dirs.shape[0]).astype(np.int16);
    i = np.zeros_like(base_probe_idx, dtype=np.uint16)
    ray_dirs[:, NEAREST_PROBE_IDX] = i
    ray_dirs[:, RELATIVE_PROBE_IDX] = relative_probe_index(base_probe_idx, i, L_LAB1)
    rendered_img = np.zeros((ray_dirs.shape[0], 3)).astype(np.float32);

    count = 0
    # while count < TOTAL_NUM_PROBES:
    while count < 3:
        print(f'========================= trace {count} =============================')
        # print(f'ray_dirs[:,PROBES_LEFT]  \n {ray_dirs[:,PROBES_LEFT] }')
        # L = find_next_L(camera.eye, L_LAB1, L_LAB2, count, L_LAB3, L_LAB4)
        L = L_LAB1
        if L is None:
            break;
        # print(f'Probe {L.probeStartPosition}')
        # count += 1;
        # only trace pixeles that are not traced or (unknown and have probes left)
        indices = np.where((ray_dirs[:, PROBES_LEFT] != 0) & (ray_dirs[:, HIT_RESULT] != TRACE_RESULT_HIT)
                            # & (ray_dirs[:, HIT_RESULT] != TRACE_RESULT_MISS) # TODO: comment out
                            )[0]

        if ray_dirs[indices].shape[0] == 0:
            # print('stop as no probes left for all ray')
            return rendered_img, ray_dirs;

        # ======================== trace_one_probe_oct ==============================
        # temp_time = time.time()
        # print(f'time before trace_one_probe:', temp_time - start, " s")
        trace_res = trace_one_probe_oct(origin, ray_dirs[indices], L); # update ray dir, eg. reuslt, hit coordiates
        # print(f'trace_res \n {trace_res[:, HIT_RESULT]}')

        # tracing results using the second probe
        sub_rendered_img_L_next = None
        # not_hit_indices_L_next = np.arange(ray_dirs.shape[0])
        hit_indices_L_next = None

        ray_dirs[indices] = trace_res

        # CASE1: trace result unknown
        # i = -1 # no next probe to trace, probes_left -= 1;
        # unknown_indices = np.where(ray_dirs[:, HIT_RESULT] == TRACE_RESULT_UNKNOWN)[0]
        # TODO: comment out above code
        unknown_indices = np.where(ray_dirs[:, HIT_RESULT] != TRACE_RESULT_HIT)[0]
        new_unknown = ray_dirs[unknown_indices]
        new_unknown[:, NEAREST_PROBE_IDX] = -1 # i
        new_unknown[:, PROBES_LEFT] -= 1 # probes left

        ray_dirs[unknown_indices] = new_unknown
        i_unknown = next_cycle_index(i[unknown_indices]); # 0-7
        if i_unknown.size != 0:
            i[unknown_indices] = i_unknown
            ray_dirs[unknown_indices, NEAREST_PROBE_IDX] = i_unknown
            ray_dirs[unknown_indices, RELATIVE_PROBE_IDX] = relative_probe_index(
                                                              base_probe_idx[unknown_indices],
                                                              i_unknown, L)
            ray_dirs[unknown_indices, HIT_RESULT] = np.ones(unknown_indices[0].size) * -1
        # case2: trace result hit
        # ===================== debug =========================================================
        # if SHOW_SINGLE_PROBE_TRACE and (indices.size - unknown_indices.size > 0):
        # # display pixels hit in this round
        hit_indices = np.where(trace_res[:, HIT_RESULT] == TRACE_RESULT_HIT)[0]
        hitted_ray = trace_res[hit_indices];
        hit_indices = indices[hit_indices]
        # print(f'hit_indices {hit_indices.size}')

        # print(f'\n\n====================== trace by probes ======================')
        # print(f'this round HIT {indices.size - unknown_indices.size} \n')
        end = time.time()
        print(f'rendering time:', end - start, " s")

        sub_rendered_img = visualize_hit_pixel(ray_dirs.shape[0], hitted_ray, hit_indices, L, SHOW=True)
 
        rendered_img[indices] = sub_rendered_img[indices]
        count += 1

    return rendered_img, ray_dirs

def visualize_hit_pixel(total_pixel, hitted_ray, hit_indices, L, SHOW=True):
    """display pixels that are hitted"""
    rendered_img = np.zeros((total_pixel, 3)).astype(np.float32);

    hit_tex_coord = hitted_ray[:, HIT_PROBE_TEX_X:HIT_PROBE_TEX_Y+1]
    probe_idx = hitted_ray[:, RELATIVE_PROBE_IDX].astype(int)
    # print(f'L.radianceProbeGrid {L.radianceProbeGrid[0].shape}')
    radiance = texel_fetch(hit_tex_coord, 'radiance', probe_idx, L.radianceProbeGrid)
    rendered_img[hit_indices] = radiance

    # show hit pixels this round
    if SHOW:
        plt.imshow(np.clip(rendered_img.reshape(-1, int(np.sqrt(total_pixel)), 3).get(), 0, 1))
        plt.show()
    return rendered_img

def ray_dirs_with_params(ray_dir, num_pixels):
    params = np.zeros((num_pixels, 8))
    params[:, 1] = HIT_DISTANCE # t_max
    params[:, 2:4] = np.inf # hit probe tex coord x y
    params[:, 4] = -1 # relative index
    params[:, 5] = TOTAL_NUM_PROBES
    params[:, 6:] = -1 # hit result


    ray_dirs = ray_dir.reshape(num_pixels, num_channel)
    ray_dirs = np.c_[ray_dirs, params]
    return ray_dirs

def render(ray_origin, ray_dir, size, L, L2=None, L3=None, L4=None):
    row, col, num_channel = ray_dir.shape
    num_pixels = row * col
    ray_dirs = ray_dirs_with_params(ray_dir, num_pixels)

    tracable_ray_indices = None
    # rendered_img = np.zeros((num_pixels, 3)).astype(np.float32);

    # temp_time = time.time()
    # print(f'time before trace:', temp_time - start, " s")
    rendered_img, ray_dirs = trace(ray_origin, ray_dirs, None, L, L2, L3, L4)

    if DEBUG:
        print(f'ray_dirs HIT_PROBE_TEX \n {ray_dirs[:, HIT_PROBE_TEX_X:HIT_PROBE_TEX_Y+1]}')

    hit_indices = np.where(ray_dirs[:, HIT_RESULT] == TRACE_RESULT_HIT)[0]
    not_hit_indices = np.where(ray_dirs[:, HIT_RESULT] != TRACE_RESULT_HIT)[0]

    return rendered_img, not_hit_indices, hit_indices

# render and intermediate params 220 + texel fetch 84mb + trace 837mb = 1141mb
def reconstruct_scene(size, base, camera, L, save=False, L2=None, L3=None, L4=None):
    input_ray_dirs = getPixelCoordsAndRayDir(camera, size, size)
    # print(f'input_ray_dirs {input_ray_dirs.shape}')
    input_ray_dirs = input_ray_dirs[None]
    rendered_img = np.zeros((size * size, 3)).astype(np.float32);
    prev_not_hit_idx = None
    count = 0
    trace_num = size * size if prev_not_hit_idx is None else prev_not_hit_idx.size
    row_buffer, not_hit_idx_sub, hit_idx_sub = render(camera.eye, input_ray_dirs.reshape(1, trace_num, 3), size, \
                                                          L, L2, L3, L4)
    if prev_not_hit_idx is None:
        prev_not_hit_idx = not_hit_idx_sub
        idx = hit_idx_sub
    else:
        idx = prev_not_hit_idx[hit_idx_sub]
        prev_not_hit_idx = prev_not_hit_idx[not_hit_idx_sub]

    rendered_img[idx] = row_buffer[hit_idx_sub]
    # num_pixels_to_show = END-START if DEBUG else size
    # # print(f'rendered_img {rendered_img.shape}')
    # plt.imshow(np.clip(rendered_img.reshape(-1, int(num_pixels_to_show), 3).get(), 0, 1))
    # plt.show()

    # print(f'not_hit_indices {not_hit_indices.size} \n {not_hit_indices}')
    input_ray_dirs = getPixelCoordsAndRayDir(camera, size, size, jitter=True)

    input_ray_dirs = input_ray_dirs[prev_not_hit_idx]
    return rendered_img

def createLightFieldProbeSurface(pkfile, PROBE_POS, DIVISION=255):
    with open(pkfile, 'rb') as inp:
            tex2dArr = pickle.load(inp)
    # tex2dArr = createTexture2DArrays(num_probes, path, DIVISION);

    probe_info = get_probe_info(PROBE_POS);

    light_field_surface = LightFieldSurface(
        radianceProbeGrid=np.asarray(tex2dArr['radiance']),
        normalProbeGrid=np.asarray(tex2dArr['normals']),
        distanceProbeGrid=np.asarray(tex2dArr['distance']),
        lowResolutionDistanceProbeGrid=np.asarray(tex2dArr['distanceLow']),
        # lowResolutionNormalProbeGrid=tex2dArr['normalLow'],
        probeCounts=probe_info['probeCounts'],
        probeStartPosition=probe_info['probeStartPosition'],
        probeStep=probe_info['probeStep'],
        lowResolutionDownsampleFactor=16,
    )
    return light_field_surface

    

if __name__=="__main__": 
    # install('chainer')
    # with open(f'/content/L_LAB1_32.pkl', 'rb') as inp:
    #     L_LAB1 = pickle.load(inp)
    load_probe_time_start = time.time()

    pkfile = 'tex2dArr_LAB1_32.pkl'
    PROBE_POS1 = np.array([-0.5,  4.,  -0. ])
    PROBE_COUNTS = np.array([1, 1, 1])
    PROBE_STEP = np.array([0.5, 0.5, 0.5])

    L_LAB1 = createLightFieldProbeSurface(pkfile, PROBE_POS1)
    load_probe_time_end = time.time()
    print(f'load probe time:', load_probe_time_end - load_probe_time_start, " s\n\n")



    # at_scanner = np.array([0.956, -3.563, 3.591]) # 7probes probe3
    at_scanner = np.array([0.956, -3.563, 3.591]) # 7probes probe1

    at = scanner_to_cg(at_scanner)

    eye = L_LAB1.probeStartPosition
    # eye = np.array([ 0.5,   4.5, -0.  ])

    FOV = 70
    UP = np.array([0, 1, 0])
    NUM_PROBES = 1;
    BILINEAR_INTERPOLATION = True
    distance_degeneration = 0
    probe_origin_degenerate = np.zeros(3)

    eye_arr = [eye]
    at_arr = [at]
    size = 128
    base = 128
    DEBUG = False
    BLEND = False
    GAMMA= False
    
    # print(f'main')
    start = time.time()
    camera = Camera(eye_arr[0], at, UP, FOV)

    img_arr = reconstruct_scene(size, base, camera, L_LAB1, save=False, \
                                L2=None, L3=None, L4=None)
    end = time.time()
    

    print(f'rendering time:', end - start, " s")




	# start = timer() 
	# func(a) 
	# print("without GPU:", timer()-start)	 
	