import pdb
import cv2
import os
import torch
import numpy as np
from os.path import join
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ..core import constants
from ..core.config import DATASET_FILES, DATASET_FOLDERS, SMPL_MODEL_DIR, \
                          JOINT_REGRESSOR_H36M, DATASET_NPZ_PATH
from ..utils.image_utils import crop_cv2, flip_img, flip_pose, flip_kp, transform, rot_aa
from ..utils.geometry import batch_rot2aa, batch_rodrigues, estimate_translation_np
from ..models import SMPL

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(
        self, options, dataset, 
        use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=False, use_augmentation=True, 
        is_train=True, num_images=0
    ):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        ds_file = join(DATASET_NPZ_PATH, DATASET_FILES[is_train][dataset])
        logger.info(f'Loading npz file from {ds_file}...')
        self.data = np.load(ds_file)
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
        self.imgname = self.data['imgname']

        # activate follwing data
        self.use_smpl = use_smpl
        self.use_pose3d = use_pose3d
        self.use_pose2d = use_pose2d
        self.use_surface2d = use_surface2d

        if num_images > 0 and is_train:
            # select a random subset of the dataset
            rand = np.random.randint(0, len(self.imgname), size=(num_images))
            logger.info(f'{rand.shape[0]} images are randomly sampled from {self.dataset}')
            self.imgname = self.imgname[rand]
            self.data_subset = {}
            for f in self.data.files:
                self.data_subset[f] = self.data[f][rand]
            self.data = self.data_subset

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(float)
            self.betas = self.data['shape'].astype(float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl'].astype(bool)
            else:
                self.has_smpl = np.ones(len(self.imgname)).astype(bool)
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname)).astype(bool)
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = True
        except KeyError:
            self.has_pose_3d = False
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
            self.has_pose_2d = True
        except KeyError:
            try:
                mmpose_keypoints = self.data['mmpose_keypoints']
                joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
                keypoints_gt = np.zeros([self.imgname.shape[0], 24,3])
                keypoints_gt[:, joints_idx, :] = mmpose_keypoints[:, :17, :]
            except KeyError:
                keypoints_gt = np.zeros((len(self.imgname), 24, 3))
                self.has_pose_2d = False
        try:
            keypoints_openpose = self.data['openpose']
            self.has_pose_2d = True
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(int)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(int)
        
        # evaluation variables
        if not self.is_train:
            self.joint_mapper_h36m = constants.H36M_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()

            self.smpl_male = SMPL(SMPL_MODEL_DIR, gender='male', create_transl=False)
            self.smpl_female = SMPL(SMPL_MODEL_DIR, gender='female', create_transl=False)

        self.length = self.scale.shape[0]
        logger.info(f'Loaded {self.dataset} dataset, num samples {self.length}')
        
    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train and self.use_augmentation == True:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.NOISE_FACTOR, 1+self.options.NOISE_FACTOR, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.ROT_FACTOR,
                    max(-2*self.options.ROT_FACTOR, np.random.randn()*self.options.ROT_FACTOR))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.SCALE_FACTOR,
                    max(1-self.options.SCALE_FACTOR, np.random.randn()*self.options.SCALE_FACTOR+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, kp2d=None):
        """Process rgb image and do augmentation."""
        rgb_img = crop_cv2(rgb_img, center, scale,
                  [self.options.IMG_RES, self.options.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2] + 1, center, scale,
                                  [self.options.IMG_RES, self.options.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2. *kp[:,:-1] / self.options.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = join(self.img_dir, self.imgname[index])

        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(float)
        except TypeError:
            logger.error(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.j2d_processing(keypoints, center, sc * scale, rot, flip)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn, keypoints)
        img = torch.from_numpy(img).float()

        # Store image before normalization to use it in visualization
        item['index'] = index

        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        item['keypoints'] = torch.from_numpy(keypoints).float()

        item['has_smpl'] = np.array(self.has_smpl[index] & self.use_smpl).astype(bool)
        item['has_pose_3d'] = np.array(self.has_pose_3d & self.use_pose3d).astype(bool)
        item['has_pose_2d'] = np.array(self.has_pose_2d & self.use_pose2d).astype(bool)
        item['has_surface_2d'] = np.array(self.has_smpl[index] & self.use_surface2d).astype(bool)
        item['scale'] = np.array(sc * scale).astype(float)
        item['center'] = np.array(center).astype(float)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.array(rot).astype(float)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        # prepare pose_3d for evaluation
        # For 3DPW get the 14 common joints from the rendered shape
        if not self.is_train:
            if self.dataset in ['3dpw', '3dpw-all','ochuman', 'lspet', '3doh']:
                if self.options.GENDER_EVAL == True:
                    # pdb.set_trace()
                    gt_vertices = self.smpl_male(global_orient=item['pose'].unsqueeze(0)[:,:3], 
                                                 body_pose=item['pose'].unsqueeze(0)[:,3:], 
                                                 betas=item['betas'].unsqueeze(0)).vertices
                    gt_vertices_f = self.smpl_female(global_orient=item['pose'].unsqueeze(0)[:,:3], 
                                                     body_pose=item['pose'].unsqueeze(0)[:,3:], 
                                                     betas=item['betas'].unsqueeze(0)).vertices 
                    gt_vertices = gt_vertices if item['gender'] == 0 else gt_vertices_f
                else:
                    gt_vertices = self.smpl(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    ).vertices

                J_regressor_batch = self.J_regressor[None, :].expand(1, -1, -1)
                pose_3d = torch.matmul(J_regressor_batch, gt_vertices)
                pelvis = pose_3d[:, [0], :].clone()
                pose_3d = pose_3d[:, self.joint_mapper_h36m, :]
                pose_3d = pose_3d - pelvis
                item['pose_3d'] = pose_3d[0].float()
                item['vertices'] = gt_vertices[0].float()
            else:
                item['pose_3d'] = item['pose_3d'][self.joint_mapper_gt, :-1].float()

        return item

    def __len__(self):
        return len(self.imgname)
