import torch
import torch.nn.functional as F
import kornia

__all__ = [
    "add_film_grain",
    "match_color_to_reference",
    "unsharp_sharpen",
    "laplacian_sharpen",
    "sobel_sharpen",
    "VideoEnhancer",
]

def add_film_grain(
    images, grain_intensity=0.04, saturation_mix=0.5
):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    device = images.device
    grain = torch.randn_like(images)
    grain[..., 0] *= 2.0  # red
    grain[..., 2] *= 3.0  # blue
    gray_grain = grain[..., 1].unsqueeze(-1).expand_as(grain)
    grain = saturation_mix * grain + (1.0 - saturation_mix) * gray_grain
    return torch.clamp(images + grain * grain_intensity, 0.0, 1.0)

def match_color_to_reference(
    images, reference_image, match_strength=1.0
):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if reference_image.ndim == 3:
        reference_image = reference_image.unsqueeze(0)
    device = images.device
    img_nchw = images.permute(0, 3, 1, 2).to(device)
    ref_nchw = reference_image.permute(0, 3, 1, 2).to(device)
    img_lab = kornia.color.rgb_to_lab(img_nchw)
    ref_lab = kornia.color.rgb_to_lab(ref_nchw)
    img_mean = img_lab.mean([2, 3], keepdim=True)
    img_std = img_lab.std([2, 3], keepdim=True) + 1e-5
    ref_mean = ref_lab.mean([2, 3], keepdim=True)
    ref_std = ref_lab.std([2, 3], keepdim=True)
    matched = (img_lab - img_mean) / img_std * ref_std + ref_mean
    blended = match_strength * matched + (1.0 - match_strength) * img_lab
    return torch.clamp(kornia.color.lab_to_rgb(blended), 0.0, 1.0).permute(0, 2, 3, 1)

def unsharp_sharpen(images, strength=0.5):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    device = images.device
    x = images.permute(0, 3, 1, 2).to(device)
    blurred = F.avg_pool2d(x, 3, stride=1, padding=1)
    sharpened = x + strength * (x - blurred)
    return torch.clamp(sharpened, 0.0, 1.0).permute(0, 2, 3, 1)

def laplacian_sharpen(images, strength=0.5):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    device = images.device
    x = images.permute(0, 3, 1, 2).to(device)
    kernel = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32, device=device).expand(3,1,3,3)
    edges = F.conv2d(x, kernel, padding=1, groups=3)
    sharpened = x + strength * edges
    return torch.clamp(sharpened, 0.0, 1.0).permute(0, 2, 3, 1)

def sobel_sharpen(images, strength=0.5):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    device = images.device
    x = images.permute(0, 3, 1, 2).to(device)
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).expand(3,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).expand(3,1,3,3)
    gx = F.conv2d(x, sx, padding=1, groups=3)
    gy = F.conv2d(x, sy, padding=1, groups=3)
    edges = (gx ** 2 + gy ** 2 + 1e-6).sqrt()
    sharpened = x + strength * edges
    return torch.clamp(sharpened, 0.0, 1.0).permute(0, 2, 3, 1)

class VideoEnhancer:
    @staticmethod
    def apply_grain(frame, **kwargs):
        return add_film_grain(frame, **kwargs)

    @staticmethod
    def color_match(frame, ref, **kwargs):
        return match_color_to_reference(frame, ref, **kwargs)

    @staticmethod
    def sharpen_unsharp(frame, **kwargs):
        return unsharp_sharpen(frame, **kwargs)

    @staticmethod
    def sharpen_laplacian(frame, **kwargs):
        return laplacian_sharpen(frame, **kwargs)

    @staticmethod
    def sharpen_sobel(frame, **kwargs):
        return sobel_sharpen(frame, **kwargs)

    @classmethod
    def enhance_frame(cls, frame, reference=None, settings=None):
        cfg = settings or {}
        out = frame
        if 'sharpen' in cfg:
            m = cfg['sharpen'].get('method', 'unsharp')
            s = cfg['sharpen'].get('strength', 0.5)
            out = getattr(cls, f"sharpen_{m}")(out, strength=s)
        if cfg.get('grain', {}).get('enabled', True):
            out = cls.apply_grain(
                out,
                grain_intensity=cfg['grain'].get('intensity', 0.04),
                saturation_mix=cfg['grain'].get('saturation', 0.5)
            )
        if reference is not None and 'color_match' in cfg:
            out = cls.color_match(out, reference, match_strength=cfg['color_match'].get('strength', 1.0))
        return out
