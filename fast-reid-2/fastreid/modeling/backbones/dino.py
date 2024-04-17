from functools import partial
import logging
import torch

from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)

@BACKBONE_REGISTRY.register()
def build_dino_backbone(cfg):
    """
    Create a DINO V2 instance from config.
    Returns:
        nn.Module class instance
    """
    # fmt: off
    input_size      = cfg.INPUT.SIZE_TRAIN
    BACKBONE_SIZE   = cfg.MODEL.BACKBONE.BACKBONE_SIZE #("small", "base", "large" or "giant")
    # fmt: on

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    logger.info(f"Loading dinov2 {BACKBONE_SIZE}: {backbone_name} model")

    model = torch.hub.load("facebookresearch/dinov2", model=backbone_name)
    origin_dinov2_class = type(model)

    del model
    torch.cuda.empty_cache()  

    class DINOv2(origin_dinov2_class):
        def __init__(self, *args, **kwargs):
            super(origin_dinov2_class, self).__init__(*args, **kwargs)
            self.model = torch.hub.load("facebookresearch/dinov2", model=backbone_name)
            self.model.forward = partial(
            self.model.get_intermediate_layers,
            # n=cfg.model.backbone.out_indices,
            reshape=True,
        )

        def forward(self, x):
            output = self.model(x)[0]
            return output
    
    dinov2_model = DINOv2()

    return dinov2_model
