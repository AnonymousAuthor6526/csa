import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pdb
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from loguru import logger
from einops import rearrange, reduce, repeat
from torchvision.utils import make_grid
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d, RotateAxisAngle
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    AmbientLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    BlendParams,
    SoftPhongShader,
    SoftGouraudShader,
    TexturesUV,
    TexturesVertex
)
from ...utils import kp_utils

class SoftRenderer(nn.Module):
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(
        self, 
        faces, 
        focal_length=5000., 
        img_res=224, 
        num_mask=4, 
        sigma=1e-4, 
        gamma=1e-4, 
        dist_eps=1e-4,
        use_parts_mask=False,
    ):
        super(SoftRenderer, self).__init__()
        faces = torch.from_numpy(faces.astype('int32'))
        self.faces = faces
        self.img_res = img_res
        self.num_mask = num_mask
        self.focal_length = focal_length
        self.use_parts_mask = use_parts_mask

        blend_params = BlendParams(
            background_color=torch.tensor([0.0, 0.0, 0.0],),
            sigma=sigma,
            gamma=gamma,
        )
        raster_settings = RasterizationSettings(
            image_size=img_res,
            blur_radius=0,
            # blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
            faces_per_pixel=1,
            perspective_correct=False,
        )

        # lights = PointLights(location=[[0.0, 0.0, -3.0]])
        lights = AmbientLights()

        cameras = self.get_default_camera()
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                blend_params=blend_params,
                cameras=cameras,
                lights=lights
            )
        )
        self.matrix = RotateAxisAngle(180, "Z").get_matrix()
        self.body_part_texture = self.get_body_part_texture(non_parametric=True)

        # self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.erosion = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

    def forward(
        self, 
        vertices, 
        camera_translation, 
    ):
        batch_size = vertices.shape[0]
        device = vertices.device

        verts = vertices.clone()
        cam_t = camera_translation.clone()
        # cam_t.requires_grad = False

        matrix = self.matrix.type_as(verts)
        matrix = repeat(matrix, 'b h w -> (b repeat) w h', repeat=batch_size)
        transform = Transform3d(verts.dtype, device, matrix)
        verts_rot = transform.transform_points(verts)
        batch_faces = repeat(self.faces.type_as(verts), 'n c -> b n c', b=batch_size)
        mesh = Meshes(verts=verts_rot, faces=batch_faces)

        batch_parts = repeat(self.body_part_texture.type_as(verts), 'n -> b n 3', b=batch_size)
        textures = mesh.verts_normals_padded()
        textures *= batch_parts
        mesh.textures = TexturesVertex(verts_features=textures) 

        cameras = self.get_train_camera(cam_t, batch_size, device)

        renderer = self.renderer.to(device)
        rend_img = renderer(mesh, cameras=cameras)
        rend_img = rearrange(rend_img, 'b h w c -> b c h w')

        # Losses to smooth / regularize the mesh shape
        output_dict = {}
        if self.use_parts_mask:
            # mesh body parts masks
            part_mask, part_label = self.get_parts_masks(rend_img)
            output_dict["part_mask"] = part_mask
            output_dict["part_label"] = part_label
            rend_texture = F.normalize(rend_img[:, :3], dim=1)
            rend_silhoue = rend_img[:, 3:4]
            rend_img = torch.cat([rend_texture, rend_silhoue], dim=1)
        output_dict["rend_img"] = rend_img
        return output_dict

    def get_parts_masks(self, img):
        b, c, h, w = img.shape

        textures = img[:, :3].clone().detach()
        part_label = torch.norm(textures, p=2, dim=1, keepdim=False).round().long()
        part_mask = F.one_hot(part_label, num_classes=self.num_mask + 1)
        part_mask = rearrange(part_mask, 'b h w n -> b n h w').bool()

        # dilation = -self.pool(-part_mask.float())
        # erosion = F.interpolate(dilation, size=(h, w), mode='nearest')
        # erosion = self.pool(erosion.float())

        # part_mask = -self.pool(-part_mask)
        # part_mask = self.pool(part_mask).bool()
        # self.debug_rend_out(part_mask, dilation, erosion)
        return part_mask, part_label


    def debug_rend_out(
        self,
        part_label,
        dilation, erosion, 
        idx=0
    ):
        # Helper function used for visualization in the following examples
        def identify_axes(ax_dict, fontsize=8):
            """
            Helper to identify the Axes in the examples below.

            Draws the label in a large font in the center of the Axes.

            Parameters
            ----------
            ax_dict : dict[str, Axes]
                Mapping between the title / label and the Axes.
            fontsize : int, optional
                How big the label should be.
            """
            kw = dict(fontsize=fontsize, color="black")
            for k, ax in ax_dict.items():
                ax.text(0.05, 0.9, k, transform=ax.transAxes, **kw)

        vis_img_list = []

        disp_img = rearrange(part_label[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(dilation[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)
        disp_img = rearrange(erosion[idx].detach().clone().cpu().float(), 'n h w -> h (n w)')
        vis_img_list.append(disp_img)

        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["part_label", "dilation"],
                ["erosion", "BLANK"],
            ],
            empty_sentinel="BLANK",
        )
        for i, k in enumerate(ax_dict.keys()):
            ax_dict[k].imshow(vis_img_list[i])
        identify_axes(ax_dict)
        plt.show()
        plt.close('all')
        pdb.set_trace()

    def get_body_part_texture(self, n_vertices=6890, non_parametric=False):
        # (6890,)
        smpl_segmentation = joblib.load('/Data1/tianyiYue/Workspace/RS-ptn0.5/data/smpl_partSegmentation_mapping.pkl')
        smpl_vert_idx = smpl_segmentation['smpl_index']
        smpl_vert_texture = smpl_vert_idx.copy()

        if non_parametric:
            joint_mapping = eval(f'kp_utils.map_smpl_to_render_{self.num_mask}')()
            for smpl_index, common_index in joint_mapping:
                for i in smpl_index:
                    smpl_vert_texture[smpl_vert_idx==i] = common_index

        smpl_vert_texture = (smpl_vert_texture + 1)
        return torch.from_numpy(smpl_vert_texture).float()

    def get_default_camera(self):
        R, T = look_at_view_transform(dist=2 * self.focal_length / (self.img_res + 1e-9), elev=0, azim=0)
        cameras = OpenGLPerspectiveCameras(R=R, T=T)
        return cameras

    def get_train_camera(self, T, batch_size, device):
        T[:,0] *= -1.0
        T[:,1] *= -1.0
        R = repeat(torch.eye(3), 'h w -> b h w', b=batch_size)
        cameras = PerspectiveCameras(
            R=R, 
            T=T, 
            focal_length=(self.focal_length,), 
            principal_point=((self.img_res // 2, self.img_res // 2),) ,
            in_ndc=False, 
            image_size=((self.img_res, self.img_res),), 
            device=device
        )
        return cameras