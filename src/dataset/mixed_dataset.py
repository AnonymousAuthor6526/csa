"""
This file contains the definition of different heterogeneous datasets used for training
"""
import pdb
import torch
import numpy as np

from loguru import logger
from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, db_set, **kwargs):

        length_itw =0


        if db_set == 'h36m':       
            
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'h36m']
            self.dataset_dict = { 'h36m': 1}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            self.length = max([len(ds) for ds in self.datasets])

            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'cocoall3d_h36m':       
            
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]

            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[0:1]])
            self.length = max([len(ds) for ds in self.datasets])

            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco3d_h36m_inf':       
            
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'h36m', 'mpi-inf-3dhp']
            self.dataset_dict = { 'coco3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2}
            self.partition = [0.3, 0.5, 0.2]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=False, **kwargs),
                BaseDataset(options, 'h36m', use_smpl=False, use_pose3d=True, use_pose2d=True, use_surface2d=False, **kwargs),
                BaseDataset(options, 'mpi-inf-3dhp', use_smpl=False, use_pose3d=True, use_pose2d=True, use_surface2d=False, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[0:1]])
            self.length = max([len(ds) for ds in self.datasets])

            logger.info(f"sampling rate: {self.partition}") 

        
        elif db_set == 'cocoall3d_h36m_inf':
            
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'cocoall3d', 'h36m', 'mpi-inf-3dhp']
            self.dataset_dict = { 'cocoall3d': 0, 'h36m': 1, 'mpi-inf-3dhp':2}
            self.partition = [0.4, 0.5, 0.1]

            self.datasets = [
                BaseDataset(options, 'cocoall3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
                BaseDataset(options, 'h36m', use_smpl=False, use_pose3d=True, use_pose2d=False, use_surface2d=False, **kwargs),
                BaseDataset(options, 'mpi-inf-3dhp', use_smpl=False, use_pose3d=True, use_pose2d=True, use_surface2d=False, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[0:1]])
            self.length = max([len(ds) for ds in self.datasets])

            logger.info(f"sampling rate: {self.partition}")


        elif db_set == 'cocoall3d_h36m_inf_3dpw':       
            
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'h36m', 'mpi-inf-3dhp', 'cocoall3d', '3dpw']
            self.dataset_dict = { 'h36m': 0, 'mpi-inf-3dhp':1, 'cocoall3d': 2, '3dpw':3}
            self.partition = [0.4, 0.1, 0.3, 0.2]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            logger.info(f"sampling rate: {self.partition}") 


        elif db_set == 'cocoall3d_h36m_inf_mpii_lspet_lsp-orig':       
            
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'h36m', 'mpi-inf-3dhp', 'cocoall3d', 'mpii', 'lspet', 'lsp-orig']
            self.dataset_dict = { 'h36m': 0, 'mpi-inf-3dhp':1, 'cocoall3d': 2, 'mpii':3, 'lspet':4, 'lsp-orig':5}
            self.partition = [0.3, 0.1, 0.3, 0.1, 0.1, 0.1]

            self.datasets = [
                BaseDataset(options, 'h36m', use_smpl=False, use_pose3d=True, use_pose2d=True, use_surface2d=False, **kwargs),
                BaseDataset(options, 'mpi-inf-3dhp', use_smpl=False, use_pose3d=True, use_pose2d=True, use_surface2d=False, **kwargs),
                BaseDataset(options, 'cocoall3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
                BaseDataset(options, 'mpii', use_smpl=False, use_pose3d=False, use_pose2d=True, use_surface2d=False, **kwargs),
                BaseDataset(options, 'lspet', use_smpl=False, use_pose3d=False, use_pose2d=True, use_surface2d=False, **kwargs),
                BaseDataset(options, 'lsp-orig', use_smpl=False, use_pose3d=False, use_pose2d=True, use_surface2d=False, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[2:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco_h36m_inf_3dpw':       
            
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'h36m', 'mpi-inf-3dhp', 'coco', '3dpw']
            self.dataset_dict = { 'h36m': 0, 'mpi-inf-3dhp':1, 'coco': 2, '3dpw':3}
            self.partition = [0.2, 0.1, 0.3, 0.4]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])

            logger.info(f"sampling rate: {self.partition}") 

        # for ablation study

        elif db_set == 'coco3d':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d']
            self.dataset_dict = { 'coco3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'cocoall3d':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'cocoall3d']
            self.dataset_dict = { 'cocoall3d': 0}
            self.partition = [1.0]

            self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco3d_cocoall2d':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'cocoall3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoall3d': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=False, use_surface2d=False, **kwargs),
                BaseDataset(options, 'cocoall3d', use_smpl=False, use_pose3d=False, use_pose2d=True, use_surface2d=True, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco3d_cocoall2dpose':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'cocoall3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoall3d': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
                BaseDataset(options, 'cocoall3d', use_smpl=False, use_pose3d=False, use_pose2d=True, use_surface2d=False, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco3d_cocoall2dsurface':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'cocoall3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoall3d': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
                BaseDataset(options, 'cocoall3d', use_smpl=False, use_pose3d=False, use_pose2d=False, use_surface2d=True, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco3d_3dpw2dsurface':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'cocoall3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoall3d': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
                BaseDataset(options, '3dpw', use_smpl=False, use_pose3d=False, use_pose2d=False, use_surface2d=True, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco3d_3dpw2dpose':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'cocoall3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoall3d': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=False, **kwargs),
                BaseDataset(options, '3dpw', use_smpl=False, use_pose3d=False, use_pose2d=True, use_surface2d=False, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        elif db_set == 'coco3d_3dpw2dpseudo':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'cocoall3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoall3d': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
                BaseDataset(options, '3dpw', use_smpl=False, use_pose3d=False, use_pose2d=False, use_surface2d=False, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}")

        elif db_set == 'coco3d_3dpw3d':       
            logger.info(">>> Selected DBSet: {}".format(db_set))
            self.dataset_list = [ 'coco3d', 'cocoall3d']
            self.dataset_dict = { 'coco3d': 0, 'cocoall3d': 1}
            self.partition = [0.5, 0.5]

            self.datasets = [
                BaseDataset(options, 'coco3d', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
                BaseDataset(options, '3dpw', use_smpl=True, use_pose3d=True, use_pose2d=True, use_surface2d=True, **kwargs),
            ]
            total_length = sum([len(ds) for ds in self.datasets])
            length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
            self.length = max([len(ds) for ds in self.datasets])
            logger.info(f"sampling rate: {self.partition}") 

        else:
            logger.error(f"db {db_set} does not exist!") 
            exit()

        assert len(self.partition) == len(self.dataset_list)
        print(">>> Total DB num: {} | total in-the-wild DB num: {}".format(total_length, length_itw))

        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.partition)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
