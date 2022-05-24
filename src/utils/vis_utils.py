import cv2
import pdb
import torch
import joblib
import numpy as np
import skimage.io as io
import scipy.cluster.vq as scv
import skimage.transform as tr
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
from matplotlib import cm as mpl_cm, colors as mpl_colors

from . import kp_utils
from ..core.config import SMPL_MODEL_DIR
from ..models.head.smpl_head import SMPL


def show_3d_pose(kp_3d, dataset='common', radius=1, ax=None):
    if isinstance(kp_3d, torch.Tensor):
        kp_3d = kp_3d.numpy()

    if ax is None:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot('111', projection='3d', aspect='auto')

    skeleton = eval(f'kp_utils.get_{dataset}_skeleton')()
    for i, (j1, j2) in enumerate(skeleton):
        if kp_3d[j1].shape[0] == 4:
            x, y, z, v = [np.array([kp_3d[j1, c], kp_3d[j2, c]]) for c in range(4)]
        else:
            x, y, z = [np.array([kp_3d[j1, c], kp_3d[j2, c]]) for c in range(3)]
            v = [1, 1]
        ax.plot(x, y, z, lw=2, c=get_colors()['purple'] / 255)
        for j in range(2):
            if v[j] > 0: # if visible
                ax.plot(x[j], y[j], z[j], lw=2, c=get_colors()['blue'] / 255, marker='o')
            else: # nonvisible
                ax.plot(x[j], y[j], z[j], lw=2, c=get_colors()['red'] / 255, marker='x')

    hip_joint = 2
    RADIUS = radius  # space around the subject
    xroot, yroot, zroot = kp_3d[hip_joint, 0], kp_3d[hip_joint, 1], kp_3d[hip_joint, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(-90, -90)
    if ax is None:
        plt.show()


def draw_skeleton(image, kp_2d, dataset='common', unnormalize=True,
                  thickness=2, res=224, j_error=None, j_uncertainty=None, print_joints=False):

    if np.max(image) < 10:
        image = image * 255
        image = np.clip(image, 0, 255)
        image = np.asarray(image, dtype=np.uint8)

    if unnormalize:
        kp_2d[:,:2] = 0.5 * res * (kp_2d[:, :2] + 1) # normalize_2d_kp(kp_2d[:,:2], 224, inv=True)

    kp_2d = np.hstack([kp_2d, np.ones((kp_2d.shape[0], 1))])

    kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)

    rcolor = [255,0,0]
    pcolor = [0,255,0]
    lcolor = [0,0,255]

    skeleton = eval(f'kp_utils.get_{dataset}_skeleton')()
    joint_names = eval(f'kp_utils.get_{dataset}_joint_names')()

    if j_error is not None:
        cv2.putText(
            image, f'MPJPE: {j_error.mean():.1f}',
            (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0)
        )

    # common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx, pt in enumerate(kp_2d):
        # if pt[2] > 0: # if visible
        cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)
        if j_error is not None:
            cv2.putText(image, f'{j_error[idx]:.1f}', (pt[0]+3, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

        if j_uncertainty is not None:
            cv2.putText(image, f'{j_uncertainty[idx]:.6f}',
                        (pt[0]-45, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

        if print_joints:
            cv2.putText(image, f'{idx}-{joint_names[idx]}',
                        (pt[0]+3, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    for i,(j1,j2) in enumerate(skeleton):
        # if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
        # if dataset == 'common':
        #     color = rcolor if common_lr[i] == 0 else lcolor
        # else:
        color = lcolor if i % 2 == 0 else rcolor
        if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0:
            pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])
            cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    image = np.asarray(image, dtype=float) / 255.
    return image


def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0)/(2*ratio)

    return kp_2d


def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
        'pinkish': np.array([204, 77, 77]),
        'orange': np.array([244, 177, 131]),
    }
    return colors


def get_segmentation_color_map():
    mycmap = torch.Tensor([
        (161, 215, 106),
        (204, 77, 77),
        (145, 191, 219),
        (102,   153,    204),
        (102,   51,     102),
        (204,   204,    153),
        (153,   0,      51),
        (204,   255,    102),
        (255,   153,    0),
        (102,   102,    153),
        (102,   0,      51),
        (153,   204,    153),
        (102,   51,     0),
        (255,   153,    51),
        (255,   255,    102),
        (0,     102,    51),
        (102,   0,      153),
        (153,   102,    0),
        (0,     153,    51),
        (102,   102,    102),
        (204,   51,     102),
    ])
    return np.array(mycmap) / 255. # (mycmap * 255).astype(np.uint8)


def color_vertices(per_joint_label, alpha=1.0):
    """
    color vertices based on a per_joint_label, joints are native SMPL joints
    per_joint_label np.array (24,)
    alpha: transparency values
    """
    smpl_segmentation = joblib.load('data/smpl_segmentation_24joints.pkl')
    n_vertices = smpl_segmentation['smpl_index'].shape[0]

    vertex_colors = np.ones((n_vertices, 4)) * np.array([0.3, 0.3, 0.3, alpha])
    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()

    per_vertex_label = np.zeros((n_vertices))

    for idx, label in enumerate(list(per_joint_label)):
        per_vertex_label[smpl_segmentation['smpl_index'] == idx] = label

    vertex_colors[:, :3] = cm(norm_gt(per_vertex_label))[:, :3]
    return vertex_colors


def color_vertices_batch(per_joint_label, alpha=1.0):
    vertex_colors = []
    for i, j in enumerate(per_joint_label):
        vertex_colors.append(color_vertices(j, alpha=alpha))
    return np.array(vertex_colors)


def visualize_joint_error(j_error=None, res=480):
    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        global_orient=torch.from_numpy(np.array([[np.pi,0,0]])).float()
    )
    joints = smpl().joints
    joints2d = joints[:,:,:2]
    joints2d -= (joints2d[:,27,:] + joints2d[:,28,:]) / 2
    joints2d = torch.cat([joints2d, torch.ones(1, joints2d.shape[1], 1)], dim=-1)
    joints2d = joints2d[0, 25:39, :].detach().numpy()

    image = np.ones((res, res, 3)) * 255
    image = draw_skeleton(image, kp_2d=joints2d, res=res, j_error=j_error)

    # plt.imshow(image)
    # plt.show()

    return image


def visualize_joint_uncertainty(j_uncertainty=None, res=480):
    # from smplx import SMPL
    # smpl = SMPL(
    #     model_path=SMPL_MODEL_DIR,
    #     global_orient=torch.from_numpy(np.array([[np.pi,0,0]])).float()
    # )
    # joints = smpl().joints
    # joints2d = joints[:,:,:2]
    # # joints2d -= (joints2d[:,27,:] + joints2d[:,28,:]) / 2
    # joints2d = torch.cat([joints2d, torch.ones(1, joints2d.shape[1], 1)], dim=-1)
    # joints2d = joints2d[0, :24, :].detach().numpy() * 1.15
    #
    # image = np.ones((res, res, 3)) * 255
    # image = draw_skeleton(image, kp_2d=joints2d, res=res, j_uncertainty=j_uncertainty, dataset='smpl')

    image = np.ones((res, res, 3)) * 255
    joint_names = kp_utils.get_smpl_joint_names()
    for idx, jn in enumerate(joint_names):
        x = 20
        y = 20 + idx * 18
        cv2.putText(image, f'{jn}: {j_uncertainty[idx]:.10f}',
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    # plt.imshow(image)
    # plt.show()

    return image


def visualize_smpl_joint_names():
    from smplx import SMPL
    smpl = SMPL(
        model_path=SMPL_MODEL_DIR,
        global_orient=torch.from_numpy(np.array([[np.pi,0,0]])).float()
    )
    joints = smpl().joints
    joints2d = joints[:,:,:2]
    # joints2d -= (joints2d[:,27,:] + joints2d[:,28,:]) / 2
    joints2d = torch.cat([joints2d, torch.ones(1, joints2d.shape[1], 1)], dim=-1)
    joints2d = joints2d[0, :24, :].detach().numpy() * 1.15

    res = 1080

    image = np.ones((res, res, 3)) * 255
    image = draw_skeleton(image, kp_2d=joints2d, res=res, dataset='smpl', print_joints=True)
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def visualize_segmentations(image, masks, alpha: float=0.7):
    b, c, h, w = image.shape
    _, n, _, _ = masks.shape

    image = image.clone()
    masks = masks.clone()

    for i in range(n):
        mask = repeat(masks[:, i], 'b h w -> b repeat h w', repeat=c)
        color = get_segmentation_color_map()[i]
        color = rearrange(color, 'c -> 1 c 1 1')
        image = torch.where(
            mask,
            image * (1 - alpha) + alpha * color,
            image
        )

    return image

def visualize_heatmaps(image, heatmaps, mask=None, alpha=0.4, cmap='jet'):
    h, w, c = image.shape
    heatmaps = heatmaps[..., None]

    hm = tr.resize(heatmaps, (h, w), anti_aliasing=False)
    cm = mpl_cm.get_cmap(cmap)
    norm_gt = mpl_colors.Normalize()
    hm = cm(norm_gt(hm))[:,:,0,:3]
    hm = torch.from_numpy(hm).type_as(image)

    if mask is not None:
        hm_img = torch.where(
            mask,
            image * (1 - alpha) + (hm * alpha),
            image
        )
    else:
        hm_img = image * (1 - alpha) + (hm * alpha)
    return hm_img

def overlay_heatmaps(image, hm, alpha=0.4):
    hm_img = image * (1 - alpha) + (hm * alpha)
    if image.max() > 2:
        hm_img = np.clip(hm_img, 0, 255).astype(np.uint8)
    else:
        hm_img = np.clip(hm_img, 0., 1.)
    return hm_img


def colormap_to_arr(arr, cmap=mpl_cm.get_cmap('jet')):
    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    gradient = cmap(np.linspace(0.0, 1.0, 1000))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    # arr2 = arr.reshape((arr.shape[0]*arr.shape[1], arr.shape[2]))
    val = 255 if arr.max() > 2 else 1
    arr2 = np.concatenate([arr, np.ones((arr.shape[0], 1)) * val], axis=-1)
    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code, dist = scv.vq(arr2, gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values = code.astype('float') / gradient.shape[0]

    # Reshape values back to (240,240)
    # values = values.reshape(arr.shape[0], arr.shape[1])
    values = values[::-1]
    return values

def reverse_norm(images):
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
    return images