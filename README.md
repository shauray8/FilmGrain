# FilmGrain
all the post processing stuff for video/image from github.com/vrgamegirl19


#### Usage
```py
import torch
from FilmGrain import add_film_grain, VideoEnhancer

frame = torch.rand(1, 720, 1280, 3).cuda()  # [B, H, W, C]
noisy = add_film_grain(frame, grain_intensity=0.05)

enhanced = VideoEnhancer.enhance_frame(
    frame,
    settings={
        "sharpen": {"method": "unsharp", "strength": 0.8},
        "grain": {"intensity": 0.03}
    }
)
```
