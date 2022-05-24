import os
import pdb
import cv2
import json
import math
import torch
import trimesh
import joblib
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from loguru import logger
from smplx import SMPL as SMPL_native
from torch.utils.data import DataLoader

from einops import rearrange, reduce, repeat
from . import config
from . import constants
from ..models import SMPL, SoftRenderer
from ..utils.renderer import Renderer
from ..utils.eval_utils import reconstruction_error, compute_error_verts
from ..utils.dataloader import CheckpointDataLoader
from ..dataset import BaseDataset, MixedDataset
from ..utils.geometry import estimate_translation, perspective_projection, rotation_matrix_to_angle_axis

class LitModule(pl.LightningModule):
    def __init__(self, hparams):
        super(LitModule, self).__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams

        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )

        self.smpl_native = SMPL_native(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )

        self.soft_renderer = None

        if self.hparams.METHOD == 'MPQA':
            from ..models import MPQA
            self.model = MPQA(
                self.hparams.MPQA,
                img_res=self.hparams.DATASET.IMG_RES,
                pretrained=self.hparams.TRAINING.PRETRAINED,
            )
            from ..losses import MPQALoss
            self.loss_fn = MPQALoss(
                shape_loss_weight=self.hparams.MPQA.SHAPE_LOSS_WEIGHT,
                keypoint_loss_weight=self.hparams.MPQA.KEYPOINT_LOSS_WEIGHT,
                pose_loss_weight=self.hparams.MPQA.POSE_LOSS_WEIGHT,
                beta_loss_weight=self.hparams.MPQA.BETA_LOSS_WEIGHT,
                openpose_train_weight=self.hparams.MPQA.OPENPOSE_TRAIN_WEIGHT,
                gt_train_weight=self.hparams.MPQA.GT_TRAIN_WEIGHT,
                loss_weight=self.hparams.MPQA.LOSS_WEIGHT,
            )
        elif self.hparams.METHOD == 'SSPA':
            self.soft_renderer = SoftRenderer(
                faces=self.smpl.faces, 
                img_res=self.hparams.DATASET.IMG_RES,
                num_mask=self.hparams.THMR.RENDER_MASK,
                sigma=self.hparams.THMR.SIGMA_VAL, 
                gamma=self.hparams.THMR.GAMMA_VAL, 
                use_parts_mask=True,
            )
            from ..models import SSPA
            self.model = SSPA(
                self.hparams,
                backbone=self.hparams.SSPA.BACKBONE,
                regressor=self.hparams.SSPA.REGRESSOR,
                num_part=11,
                img_res=self.hparams.DATASET.IMG_RES,
            )
            from ..losses import SSPA
            self.loss_fn = SSPALoss(
                hparams=self.hparams,
                shape_loss_weight=self.hparams.SSPA.SHAPE_LOSS_WEIGHT,
                keypoint_loss_weight=self.hparams.SSPA.KEYPOINT_LOSS_WEIGHT,
                pose_loss_weight=self.hparams.SSPA.POSE_LOSS_WEIGHT,
                beta_loss_weight=self.hparams.SSPA.BETA_LOSS_WEIGHT,
                openpose_train_weight=self.hparams.SSPA.OPENPOSE_TRAIN_WEIGHT,
                gt_train_weight=self.hparams.SSPA.GT_TRAIN_WEIGHT,
                loss_weight=self.hparams.SSPA.LOSS_WEIGHT,
            )
        else:
            logger.error(f'{self.hparams.METHOD} is undefined!')
            exit()

        self.renderer = Renderer(
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_res=self.hparams.DATASET.IMG_RES,
            faces=self.smpl.faces,
            mesh_color=self.hparams.DATASET.MESH_COLOR,
        )

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        self.init_evaluation_variables()

        self.pl_logging = self.hparams.PL_LOGGING

    def init_evaluation_variables(self):
        # Initialiatize variables related to evaluation
        ds_list = self.hparams.DATASET.VAL_DS.split('_')
        self.best_result = {ds:math.inf for ds in ds_list}
        self.best_pampjpe = {ds:math.inf for ds in ds_list}
        self.best_mpjpe = {ds:math.inf for ds in ds_list}
        self.best_v2v = {ds:math.inf for ds in ds_list}
        self.val_accuracy_results = {ds:[] for ds in ds_list}

        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.val_mpjpe = [] # np.zeros(len(self.val_ds))
        self.val_pampjpe = [] # np.zeros(len(self.val_ds))
        self.val_v2v = []

        # This dict is used to store metrics and metadata for a more detailed analysis
        # per-joint, per-sequence, occluded-sequences etc.
        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'mpjpe': [], # np.zeros((len(self.val_ds), 14)),
            'pampjpe': [], # np.zeros((len(self.val_ds), 14)),
            'v2v': [],
        }

        if self.hparams.TESTING.SAVE_RESULTS:
            self.evaluation_results['pose'] = []
            self.evaluation_results['shape'] = []
            self.evaluation_results['cam_t'] = []
            self.evaluation_results['vertices'] = []

    def reset_evaluation_variables(self):
        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.val_mpjpe = [] # np.zeros(len(self.val_ds))
        self.val_pampjpe = [] # np.zeros(len(self.val_ds))
        self.val_v2v = []

        # This dict is used to store metrics and metadata for a more detailed analysis
        # per-joint, per-sequence, occluded-sequences etc.
        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'mpjpe': [], # np.zeros((len(self.val_ds), 14)),
            'pampjpe': [], # np.zeros((len(self.val_ds), 14)),
            'v2v': [],
        }

        if self.hparams.TESTING.SAVE_RESULTS:
            self.evaluation_results['pose'] = []
            self.evaluation_results['shape'] = []
            self.evaluation_results['cam_t'] = []
            self.evaluation_results['vertices'] = []

    def forward(self):
        return None

    def training_step(self, batch, batch_nb):
        # Get data from the batch
        inputs = batch['img']

        # get ground truth smpl vertices and camera
        gt_out = self.smpl(
            betas=batch['betas'],
            body_pose=batch['pose'][:, 3:],
            global_orient=batch['pose'][:, :3]
        )
        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_model_joints = gt_out.joints
        gt_model_joints[batch['has_pose_3d'], 25:] = batch['pose_3d'][batch['has_pose_3d'],:,:3]
        gt_keypoints_2d_orig = batch['keypoints'].clone()
        gt_keypoints_2d_orig[..., :-1] = 0.5 * self.hparams.DATASET.IMG_RES * (gt_keypoints_2d_orig[..., :-1] + 1)
        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=self.hparams.DATASET.FOCAL_LENGTH,
            img_size=self.hparams.DATASET.IMG_RES,
        )
        batch['gt_cam_t'] = gt_cam_t
        batch['vertices'] = gt_out.vertices

        if self.training and self.soft_renderer is not None: 
            has_surface_2d = batch['has_surface_2d']
            b, _, h, w = inputs.shape
            n = self.soft_renderer.num_mask + 1

            gt_rgbo = torch.zeros([b, 4, h, w]).type_as(inputs)
            gt_al = torch.zeros([b, n, h, w]).type_as(inputs).bool()
            gt_al_label = torch.zeros([b, h, w]).type_as(inputs).long()

            if torch.sum(has_surface_2d) > 0:
                render_dict = self.soft_renderer(
                    vertices=batch['vertices'][has_surface_2d], 
                    camera_translation=batch['gt_cam_t'][has_surface_2d],
                )
                gt_rgbo[has_surface_2d] = render_dict["rend_img"]
                gt_al[has_surface_2d] = render_dict["part_mask"]
                gt_al_label[has_surface_2d] = render_dict["part_label"]

            batch.update({
                'gt_rgbo': gt_rgbo,
                'gt_al': gt_al,
                'gt_al_label': gt_al_label,
            })
            # Forward pass
            pred = self.model(inputs, gt_rgbo=batch["gt_rgbo"], gt_al=batch["gt_al"], has_gt=has_surface_2d)
        else:
            # Forward pass
            pred = self.model(inputs)

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)

        if batch_nb % self.hparams.TRAINING.LOG_FREQ_TB_IMAGES == 0:
            self.train_summaries(input_batch=batch, output=pred)

        self.log_dict(
            loss_dict, 
            logger=True, 
            sync_dist=True, 
            rank_zero_only=True
        )
        return {'loss': loss}

    def train_summaries(self, input_batch, output, nb_max_img=2):
        images = input_batch['img']
        b, _, _, _ = images.shape
        nb_max_img = min(b, nb_max_img)
        idx = torch.randperm(b)[:nb_max_img]

        pred_vertices = output['smpl_vertices'].detach()
        opt_vertices = input_batch['vertices']

        pred_cam_t = output['pred_cam_t'].detach()
        opt_cam_t = input_batch['gt_cam_t']

        pred_kp_2d = output['smpl_joints2d'][:, 25:, :].detach()
        gt_kp_2d = input_batch['keypoints'][:, 25:, :]

        online_rgbo = output['online_rgbo'].detach() if 'online_rgbo' in output.keys() else None
        target_rgba = output['target_rgba'].detach() if 'target_rgba' in output.keys() else None
        gt_rgbo = input_batch['gt_rgbo'] if 'gt_rgbo' in input_batch.keys() else None

        online_mask = output['online_al'].detach() if 'online_al' in output.keys() else None
        target_mask = output['target_al'].detach() if 'target_al' in output.keys() else None
        gt_mask = input_batch['gt_al'] if 'gt_al' in input_batch.keys() else None

        images_pred = self.renderer.visualize_tb(
            vertices=pred_vertices, camera_translation=pred_cam_t, images=images,
            kp_2d=pred_kp_2d, 
            online_mask=online_mask, target_mask=target_mask, gt_mask=None,
            online_rgbo=online_rgbo, target_rgba=target_rgba, gt_rgbo=None,
            sideview=self.hparams.TESTING.SIDEVIEW,
            idx=idx,
        )
        images_opt = self.renderer.visualize_tb(
            vertices=opt_vertices, camera_translation=opt_cam_t, images=images,
            kp_2d=gt_kp_2d,
            online_mask=None, target_mask=None, gt_mask=gt_mask,
            online_rgbo=None, target_rgba=None, gt_rgbo=gt_rgbo,
            sideview=self.hparams.TESTING.SIDEVIEW,
            idx=idx,
        )

        if self.pl_logging == True:
            self.logger.experiment.add_image('pred_shape', images_pred, self.global_step)
            self.logger.experiment.add_image('opt_shape', images_opt, self.global_step)
            

            if 'online_visualize' in output.keys():
                attn = output['online_visualize'].detach()
                online_visualize = self.renderer.visualize_attention_tb(images, attn, idx=idx,)
                self.logger.experiment.add_image(
                    'images_attn', 
                    online_visualize, 
                    self.global_step
                )

            if 'energy_visual' in output.keys():
                self.logger.experiment.add_image(
                    'pseudo/energy_visual', 
                    output['energy_visual'], 
                    self.global_step
                )

        if self.hparams.TRAINING.SAVE_IMAGES == True:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)
            save_dir = os.path.join(self.hparams.LOG_DIR, 'train_output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                    os.path.join(save_dir, f'result_{self.global_step:05d}.png'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
            )

    def validation_step(self, batch, batch_nb, dataloader_nb=0):
        images = batch['img']
        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        curr_batch_size = images.shape[0]

        with torch.no_grad():
            pred = self.model(images)
            pred_vertices = pred['smpl_vertices']

        joint_mapper_h36m = constants.H36M_TO_J17 if dataset_names[0] == 'mpi-inf-3dhp' \
            else constants.H36M_TO_J14

        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

        gt_keypoints_3d = batch['pose_3d'].cuda()
        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        idx_start = batch_nb * self.hparams.DATASET.BATCH_SIZE
        idx_stop = batch_nb * self.hparams.DATASET.BATCH_SIZE + curr_batch_size

        # Reconstuction_error
        r_error, r_error_per_joint = reconstruction_error(
            pred_keypoints_3d.cpu().numpy(),
            gt_keypoints_3d.cpu().numpy(),
            reduction=None,
        )

        # Per-vertex error
        if 'vertices' in batch.keys():
            gt_vertices = batch['vertices'].cuda()

            v2v = compute_error_verts(
                pred_verts=pred_vertices.cpu().numpy(),
                target_verts=gt_vertices.cpu().numpy(),
            )
            self.val_v2v += v2v.tolist()
        else:
            self.val_v2v += np.zeros_like(error).tolist()

        self.val_mpjpe += error.tolist()
        self.val_pampjpe += r_error.tolist()

        error_per_joint = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()

        self.evaluation_results['mpjpe'] += error_per_joint[:,:14].tolist()
        self.evaluation_results['pampjpe'] += r_error_per_joint[:,:14].tolist()

        if 'vertices' in batch.keys():
            self.evaluation_results['v2v'] += v2v.tolist()
        else:
            self.evaluation_results['v2v'] += np.zeros_like(error).tolist()

        self.evaluation_results['imgname'] += imgnames
        self.evaluation_results['dataset_name'] += dataset_names

        if self.hparams.TESTING.SAVE_RESULTS:
            tolist = lambda x: [i for i in x.cpu().numpy()]
            self.evaluation_results['pose'] += tolist(pred['pred_pose'])
            self.evaluation_results['shape'] += tolist(pred['pred_shape'])
            self.evaluation_results['cam_t'] += tolist(pred['pred_cam'])
            self.evaluation_results['vertices'] += tolist(pred_vertices)

        # this saves the rendered images
        if batch_nb % self.hparams.TESTING.LOG_FREQ_TB_IMAGES == 0:
            self.validation_summaries(batch, pred, batch_nb, dataloader_nb)

        return {
            'mpjpe': error.mean(),
            'pampjpe': r_error.mean(),
            'per_mpjpe': error_per_joint,
            'per_pampjpe': r_error_per_joint
        }

    def validation_summaries(self, input_batch, output, batch_idx, dataloader_nb=0, nb_max_img=8):
        images = input_batch['img']
        b, _, _, _ = images.shape
        nb_max_img = min(b, nb_max_img)
        # idx = torch.randperm(b)[:nb_max_img]
        idx = torch.tensor([i for i in range(nb_max_img)])

        pred_vertices = output['smpl_vertices'].detach()
        pred_cam_t = output['pred_cam_t'].detach()
        pred_kp_2d = output['smpl_joints2d'][:, 25:, :].detach()

        online_rgbo = output['online_rgbo'].detach() if 'online_rgbo' in output.keys() else None
        target_rgba = output['target_rgba'].detach() if 'target_rgba' in output.keys() else None
        gt_rgbo = input_batch['gt_rgbo'] if 'gt_rgbo' in input_batch.keys() else None

        online_mask = output['online_al'].detach() if 'online_al' in output.keys() else None
        target_mask = output['target_al'].detach() if 'target_al' in output.keys() else None
        gt_mask = input_batch['gt_al'] if 'gt_al' in input_batch.keys() else None

        images_pred = self.renderer.visualize_tb(
            vertices=pred_vertices, camera_translation=pred_cam_t, images=images,
            kp_2d=pred_kp_2d,
            online_mask=online_mask, target_mask=target_mask, gt_mask=gt_mask,
            online_rgbo=online_rgbo, target_rgba=target_rgba, gt_rgbo=gt_rgbo,
            idx=idx, sideview=self.hparams.TESTING.SIDEVIEW,
        )

        if self.pl_logging == True:
            self.logger.experiment.add_image('val_pred_shape', images_pred, self.global_step)

        if self.hparams.TESTING.SAVE_IMAGES == True:
            images_pred = images_pred.cpu().numpy().transpose(1, 2, 0) * 255
            images_pred = np.clip(images_pred, 0, 255).astype(np.uint8)
            save_dir = os.path.join(self.hparams.LOG_DIR, 'val_output_images')
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(save_dir, f'result_{dataloader_nb:02d}_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_pred, cv2.COLOR_BGR2RGB)
            )

    def validation_epoch_end(self, outputs):
        self.val_mpjpe = np.array(self.val_mpjpe)
        self.val_pampjpe = np.array(self.val_pampjpe)
        self.val_v2v = np.array(self.val_v2v)

        for k,v in self.evaluation_results.items():
            self.evaluation_results[k] = np.array(v)

        self.evaluation_results['epoch'] = self.current_epoch

        logger.info(f'***** Epoch {self.current_epoch} *****')
        for ds_idx, ds in enumerate(self.val_ds):
            ds_name = ds.dataset
            idxs = self.evaluation_results['dataset_name'] == ds_name

            mpjpe = 1000 * self.val_mpjpe[idxs].mean()
            pampjpe = 1000 * self.val_pampjpe[idxs].mean()
            v2v = 1000 * self.val_v2v[idxs].mean()

            logger.info(f'{ds_name} MPJPE: {mpjpe}')
            logger.info(f'{ds_name} PA-MPJPE: {pampjpe}')
            logger.info(f'{ds_name} V2V: {v2v}')

            acc = {
                'val_mpjpe': mpjpe.item(),
                'val_pampjpe': pampjpe.item(),
                'val_v2v': v2v.item(),
            }
            self.val_save_best_results(acc, ds_name)

            epoch_result = 1.5 * pampjpe + mpjpe

            if epoch_result < self.best_result[ds_name]:
                logger.info(f'{ds_name} Best Model Criteria Met: Current Score -> {epoch_result} \
                            | Previous Score -> {self.best_result[ds_name]}')
                self.best_result[ds_name] = epoch_result
                self.best_pampjpe[ds_name] = pampjpe
                self.best_mpjpe[ds_name] = mpjpe
                self.best_v2v[ds_name] = v2v

                # save the detailed experiment results for post-analysis script
                if self.hparams.TESTING.SAVE_RESULTS:
                    joblib.dump(
                        self.evaluation_results,
                        os.path.join(self.hparams.LOG_DIR, f'evaluation_results_{self.hparams.DATASET.VAL_DS}.pkl')
                    )
            mpjpe, pampjpe, v2v = torch.tensor(mpjpe), torch.tensor(pampjpe), torch.tensor(v2v)
            tensorboard_logs = {
                f'{ds_name}/val_loss': epoch_result, 
                f'{ds_name}/epoch/mpjpe': mpjpe, 
                f'{ds_name}/epoch/pampjpe': pampjpe,
                f'{ds_name}/epoch/v2v': v2v, 
                f'{ds_name}/best/pampjpe': self.best_pampjpe[ds_name],
                f'{ds_name}/best/mpjpe': self.best_mpjpe[ds_name],
                f'{ds_name}/best/v2v': self.best_v2v[ds_name],
            }

            self.log_dict(
                tensorboard_logs, 
                prog_bar=False, 
                logger=True,
                on_step=False, 
                on_epoch=True, 
                sync_dist=True, 
                rank_zero_only=True
            )

            # always set the first dataset as the main one
            if ds_idx == 0:
                self.log("hp_metric", epoch_result)
                self.log("val_loss", epoch_result)

        # reset evaluation variables
        self.reset_evaluation_variables()
        # return tensorboard_logs

    def test_step(self, batch, batch_nb, dataloader_nb=0):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD
        )

    def train_dataset(self):
        train_ds = MixedDataset(
            self.hparams.DATASET,
            self.hparams.DATASET.TRAIN_DS,
            num_images=self.hparams.DATASET.NUM_IMAGES,
            use_augmentation=self.hparams.TRAINING.USE_AUGM,
            is_train=True,
        )
        return train_ds

    def validation_dataset(self):
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_ds = []
        num_images = [self.hparams.DATASET.NUM_IMAGES] * len(datasets)
        for idx, dataset_name in enumerate(datasets):
            val_ds.append(
                BaseDataset(
                    options=self.hparams.DATASET,
                    dataset=dataset_name,
                    num_images=num_images[idx],
                    is_train=False,
                )
            )
        return val_ds

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        )

    def val_dataloader(self):
        self.val_ds = self.validation_dataset()
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.hparams.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                )
            )
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()

    def val_save_best_results(self, acc, ds_name):
        # log the running training metrics
        fname = f'val_accuracy_results_{ds_name}.json'
        json_file = os.path.join(self.hparams.LOG_DIR, fname)
        self.val_accuracy_results[ds_name].append([self.global_step, acc, self.current_epoch])
        with open(json_file, 'w') as f:
            json.dump(self.val_accuracy_results[ds_name], f, indent=4)