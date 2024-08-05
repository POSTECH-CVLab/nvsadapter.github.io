from typing import Dict
import gc 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from open_clip.transformer import ResidualAttentionBlock

from sgm.util import instantiate_from_config
from sgm.modules.encoders.modules import AbstractEmbModel, FrozenOpenCLIPImageEmbedder
from sgm.modules.nvsadapter.midas.api import MiDaSInference
from sgm.modules.nvsadapter.hed.api import HEDdetector
from sgm.modules.nvsadapter.canny.api import CannyInference


def zero_to_one(x):
    return (x + 1) / 2.0


def minus_one_to_one(x):
    return x * 2.0 - 1.0


def HWC3(x):
    return torch.cat([x, x, x], axis=2)


class MultipleGeneralConditioners(nn.Module):
    """
    Original GeneralConditioner concatenates the embeddings so that we cannot use
    them separately. This class allows us to use conditions separately.
    """

    def __init__(self, conditioners_config, ucg_rate=0.0):
        
        super(MultipleGeneralConditioners, self).__init__()
        conditioners, names = [], []

        for conditioner_config in conditioners_config:
            name = getattr(conditioner_config, "name", "")
            conditioner = instantiate_from_config(conditioner_config)
            conditioners.append(conditioner)
            names.append(name)

        self.conditioners = nn.ModuleList(conditioners)
        self.names = names

        embedders = []
        for conditioner in self.conditioners:
            for embedder in conditioner.embedders:
                embedders.append(embedder)
                
        self.embedders = embedders # to be compatible with GeneralConditioner
        self.ucg_rate = ucg_rate


    def __call__(self, batch: Dict):        
        conditions = dict()
        for name, conditioner in zip(self.names, self.conditioners):
            conditions.update({name + "/" + k : v for k, v in conditioner(batch).items()})

        return conditions

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        c_dict, uc_dict = dict(), dict()
        for name, conditioner in zip(self.names, self.conditioners):
            c, uc = conditioner.get_unconditional_conditioning(
                batch_c, batch_uc, force_uc_zero_embeddings
            )
            c_dict.update({name + "/" + k : v for k, v in c.items()})
            uc_dict.update({name + "/" + k : v for k, v in uc.items()})
        return c_dict, uc_dict


class ImageEmbedAttentionProjector(FrozenOpenCLIPImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder with an additional attention layer (trainable)
    """
    def __init__(self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        proj_d_model=1280,
        proj_n_head=16,
    ):
        super(ImageEmbedAttentionProjector, self).__init__(
            arch=arch,
            version=version,
            device=device, 
            max_length=max_length,
            freeze=freeze,
            antialias=antialias,
            ucg_rate=ucg_rate,
            unsqueeze_dim=unsqueeze_dim,
            repeat_to_max_len=repeat_to_max_len,
            num_image_crops=num_image_crops,
            output_tokens=output_tokens,
        )

        assert not output_tokens, "This model does not output tokens"
        assert unsqueeze_dim, "This model requires unsqueeze_dim=True to work"

        # Residual Attention layer to project the image embedding
        self.projection_layer = ResidualAttentionBlock(
            d_model=proj_d_model,
            n_head=proj_n_head,
        )
        self.model.visual.transformer = nn.Sequential(
            self.model.visual.transformer,
            self.projection_layer,
        )


class ImageEmbedLinearProjector(FrozenOpenCLIPImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder with an additional linear projection layer
    """
    def __init__(self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
        out_proj=1024,
    ):
        super(ImageEmbedLinearProjector, self).__init__(
            arch=arch,
            version=version,
            device=device, 
            max_length=max_length,
            freeze=freeze,
            antialias=antialias,
            ucg_rate=ucg_rate,
            unsqueeze_dim=unsqueeze_dim,
            repeat_to_max_len=repeat_to_max_len,
            num_image_crops=num_image_crops,
            output_tokens=output_tokens,
        )

        assert not output_tokens, "This model does not output tokens"
        assert unsqueeze_dim, "This model requires unsqueeze_dim=True to work"

        # Residual Attention layer to project the image embedding
        self.projection_layer = nn.Linear(self.model.visual.output_dim, out_proj)
        nn.init.eye_(list(self.projection_layer.parameters())[0])
        nn.init.zeros_(list(self.projection_layer.parameters())[1])

    def forward(self, image, no_dropout=False):
        feat = super().forward(image, no_dropout)
        return self.projection_layer(feat)


class RayPosConditionEmbedder(AbstractEmbModel):
    def __init__(self, offset_deg, direction_deg, use_plucker):
        """
        config
            offset_deg (List[int]): the min and max degree of ray offset vectors
            direction_deg (List[int]): the min and max degree of ray direction vectors
            use_plucker (bool): whether to use plucker coordinate
        """

        super(RayPosConditionEmbedder, self).__init__()
        self.offset_deg = offset_deg
        self.direction_deg = direction_deg
        self.use_plucker = use_plucker

        self.posemb_out_dim = (
            self.offset_deg[1] - self.offset_deg[0] + self.direction_deg[1] - self.direction_deg[0]
        ) * 6 + 6

    def rays_to_plucker(self, offset, direction):
        """
        args:
            rays_offset (torch.Tensor, [bsz, N_rays, 3]): the offset vectors of rays.
            rays_direction (torch.Tensor, [bsz, N_rays, 3]): the direction vectors of rays.
        returns:
            cross_od (torch.Tensor, [bsz, N_rays, 3]): the cross vector of the offset vectors and
                normalized direction vectors.
            normalized_d (torch.Tensor, [bsz, N_rays, 3]): the normalized direction vectors.
        """
        normalized_d = F.normalize(direction, dim=-1)
        cross_od = torch.cross(offset, normalized_d, dim=-1)
        return cross_od, normalized_d

    def sinusoidal_emb(self, x, degree):
        """
        args:
            x (torch.Tensor, [bsz, H, W, 3]): the input tensor
            degree (Tuple[int]): the tuple of min and max degree for positional encoding
        returns:
            pos_emb (torch.Tensor, [bsz, H, W, (max_deg - min_deg) * 6 + 3]): positional embedding
        """
        device = x.device
        min_deg, max_deg = degree
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device)

        scales = scales[None, None, None, None, :]
        xb = (x.unsqueeze(-1) * scales).flatten(-2, -1)
        emb = torch.sin(torch.cat([xb, xb + np.pi / 2], dim=-1))
        return torch.cat([x, emb], dim=-1)

    def forward_each(self, rays_offset, rays_direction):
        if self.use_plucker:
            # changes (o, d) -> (d/|d|, o x d)
            rays_offset, rays_direction = self.rays_to_plucker(rays_offset, rays_direction)

        # get sinusoidal embeddings for (offset/direction) & vectors.
        rays_offset_pe = self.sinusoidal_emb(rays_offset, self.offset_deg)
        rays_direction_pe = self.sinusoidal_emb(rays_direction, self.direction_deg)

        # by concatenating the PEs, get positional embs for rays.
        rays_pe = torch.cat([rays_offset_pe, rays_direction_pe], dim=-1)
        rays_pe = rearrange(rays_pe, "b n h w c -> b n c h w").contiguous()

        return rays_pe

    def forward(self, support_rays_offset, support_rays_direction, query_rays_offset, query_rays_direction):
        support_rays_pe = self.forward_each(support_rays_offset, support_rays_direction)
        query_rays_pe = self.forward_each(query_rays_offset, query_rays_direction)
        rays_pe = torch.cat([support_rays_pe, query_rays_pe], dim=1)
        return rays_pe


class ExtrinsicEmbedder(AbstractEmbModel):

    def __init__(self, deg):
        super(ExtrinsicEmbedder, self).__init__()
        self.deg = deg 

    def sinusoidal_emb(self, x, degree):
        """
        args:
            x (torch.Tensor, [bsz, H, W, 3]): the input tensor
            degree (Tuple[int]): the tuple of min and max degree for positional encoding
        returns:
            pos_emb (torch.Tensor, [bsz, H, W, (max_deg - min_deg) * 6 + 3]): positional embedding
        """
        device = x.device
        min_deg, max_deg = degree
        scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device)

        scales = scales[None, None, None, :]
        xb = (x.unsqueeze(-1) * scales).flatten(-2, -1)
        emb = torch.sin(torch.cat([xb, xb + np.pi / 2], dim=-1))
        return torch.cat([x, emb], dim=-1)

    def forward(self, support_c2ws, query_c2ws, support_latents):

        h, w = support_latents.shape[-2:]
        support_rays_pe = repeat(
            self.sinusoidal_emb(
                rearrange(
                    support_c2ws, 
                    "b n i j -> b n (i j)"
                ), self.deg
            ), "b n c -> b n c h w", h=h, w=w
        )
        query_rays_pe = repeat(
            self.sinusoidal_emb(
                rearrange(
                    query_c2ws, 
                    "b n i j -> b n (i j)"
                ), self.deg
            ), "b n c -> b n c h w", h=h, w=w
        )
        rays_pe = torch.cat([support_rays_pe, query_rays_pe], dim=1)
        return rays_pe
    

class MiDASDepthConditioner(AbstractEmbModel):

    def __init__(self, model_type):
        super(MiDASDepthConditioner, self).__init__()
        self.midas_model = MiDaSInference(model_type)

    @torch.inference_mode()
    def forward_each(self, x):
        bsz = x.shape[0]
        # midas uses [0, 1] images for inference
        x = zero_to_one(x)
        x = rearrange(x, "b n c h w -> (b n) c h w")
        # for batchfied inference
        midas_output = self.midas_model(x)
        midas_output = midas_output - midas_output.amin(dim=[1, 2, 3], keepdim=True)
        midas_output = midas_output / midas_output.amax(dim=[1, 2, 3], keepdim=True)
        # roll back the shape
        midas_output = rearrange(midas_output, "(b n) c h w -> b n c h w", b=bsz)
        torch.cuda.empty_cache()
        gc.collect()
        return midas_output
    
    @torch.inference_mode()
    def forward(self, support_rgbs_cond, query_rgbs_cond):
        support_midas_output = self.forward_each(support_rgbs_cond)
        query_midas_output = self.forward_each(query_rgbs_cond)
        midas_output = torch.cat([support_midas_output, query_midas_output], dim=1)
        ret_tensor = HWC3(midas_output).detach()
        del support_midas_output, query_midas_output, midas_output
        torch.cuda.empty_cache()
        gc.collect()
        return ret_tensor


class HEDConditioner(AbstractEmbModel):
    def __init__(self):
        super(HEDConditioner, self).__init__()
        self.hed_model = HEDdetector()

    @torch.inference_mode()
    def forward(self, support_rgbs_cond, query_rgbs_cond):
        support_hed_output = self.forward_each(support_rgbs_cond)
        query_hed_output = self.forward_each(query_rgbs_cond)
        hed_output = torch.cat([support_hed_output, query_hed_output], dim=1)
        torch.cuda.empty_cache()
        gc.collect()
        return HWC3(hed_output)
    
    @torch.inference_mode()
    def forward_each(self, x):
        bsz = x.shape[0]
        device = x.device
        # midas uses [0, 1] images for inference
        x = zero_to_one(x)
        x = rearrange(x, "b n c h w -> (b n) c h w") * 255
        # for batchfied inference
        hed_output = self.hed_model(x) / 255.
        # roll back the shape
        hed_output = torch.from_numpy(rearrange(hed_output, "(b n) c h w -> b n c h w", b=bsz)).to(device).float()
        torch.cuda.empty_cache()
        gc.collect()
        return hed_output


class CannyConditioner(AbstractEmbModel):
    def __init__(self, low_thres=100, high_thres=200):
        super(CannyConditioner, self).__init__()
        self.canny_model = CannyInference(low_thres, high_thres)

    @torch.inference_mode()
    def forward_each(self, x):
        bsz = x.shape[0]
        device = x.device
        # midas uses [0, 1] images for inference
        x = zero_to_one(x)
        x = rearrange(x, "b n c h w -> (b n) c h w").cpu() * 255
        # for batchfied inference
        canny_output = torch.from_numpy(self.canny_model(x)).to(device).float() / 255.
        # roll back the shape
        canny_output = rearrange(canny_output, "(b n) c h w -> b n c h w", b=bsz)
        ret = canny_output.detach()
        del canny_output, x
        torch.cuda.empty_cache()
        gc.collect()
        return ret
    
    @torch.inference_mode()
    def forward(self, support_rgbs_cond, query_rgbs_cond):
        support_canny_output = self.forward_each(support_rgbs_cond)
        query_canny_output = self.forward_each(query_rgbs_cond)
        canny_output = torch.cat([support_canny_output, query_canny_output], dim=1)
        torch.cuda.empty_cache()
        gc.collect()
        return HWC3(canny_output)
    

class MultipleConditioner(AbstractEmbModel):
    
    def __init__(self):
        super().__init__()
        self.conditioners = []

    @torch.inference_mode()
    def forward(self, support_rgbs_cond, query_rgbs_cond):
        ret = []
        for conditioner in self.conditioners:
            ret.append(conditioner(support_rgbs_cond, query_rgbs_cond))
        return torch.cat(ret, dim=2)

class HEDDepthConditioner(MultipleConditioner):
    def __init__(self):
        super().__init__()
        self.conditioners = nn.ModuleList([HEDConditioner(), MiDASDepthConditioner("dpt_hybrid")])

class CannyDepthConditioner(MultipleConditioner):
    def __init__(self):
        super().__init__()
        self.conditioners = nn.ModuleList([CannyConditioner(), MiDASDepthConditioner("dpt_hybrid")])

class CannyHEDConditioner(MultipleConditioner):
    def __init__(self):
        super().__init__()
        self.conditioners = nn.ModuleList([CannyConditioner(), HEDConditioner()])

class CannyHEDDepthConditioner(MultipleConditioner):
    def __init__(self):
        super().__init__()
        self.conditioners = nn.ModuleList([CannyConditioner(), HEDConditioner(), MiDASDepthConditioner("dpt_hybrid")])