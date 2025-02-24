import numpy as np
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R

def sample_transformation(r_tra, r_rot, from_frame, to_frame):
    '''
    Sample a transformation matrix with range_translation and range_rotation.
    '''
    T = RigidTransform(from_frame=from_frame, to_frame=to_frame)
    T.translation = (np.random.random(3) - 0.5) * 2 * r_tra
    rot = R.from_euler('zyx', (np.random.random(3) - 0.5) * 2 * r_rot, degrees=True)
    T.rotation = rot.as_matrix()
    return T

def rotmat_to_euler(rot):
    '''
    Convert rotation matrix to euler angles.
    '''
    return R.from_matrix(rot).as_euler('zyx', degrees=True)

def euler_to_rotmat(euler):
    '''
    Convert euler angles to rotation matrix.
    '''
    return R.from_euler('zyx', euler, degrees=True).as_matrix()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

color_style = {
    1: "{}",
    2: bcolors.OKBLUE + "{}" + bcolors.ENDC,
    3: bcolors.OKGREEN + "{}" + bcolors.ENDC,
    4: bcolors.OKCYAN + "{}" + bcolors.ENDC,
    0: bcolors.WARNING + "{}" + bcolors.ENDC,
    -1: bcolors.FAIL + "{}" + bcolors.ENDC,
    "blue": bcolors.OKBLUE + "{}" + bcolors.ENDC,
    "green": bcolors.OKGREEN + "{}" + bcolors.ENDC,
    "cyan": bcolors.OKCYAN + "{}" + bcolors.ENDC,
    "warning": bcolors.WARNING + "{}" + bcolors.ENDC,
    "red": bcolors.FAIL + "{}" + bcolors.ENDC,
    "bold": bcolors.BOLD + "{}" + bcolors.ENDC,
}

def logging(s, verbose=True, style=1):
    if not verbose:
        return
    print(color_style[style].format(s))

def calc_diff_image(img1, img2):
    diff = np.ones_like(img1, dtype=np.uint16) * 128
    diff += img1
    diff -= img2
    diff = diff.astype(np.uint8)
    return diff