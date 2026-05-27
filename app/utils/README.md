# Utils Layout

Utilities are grouped by the workflow they support:

- `assets/`: B-roll rules and external/catalog image lookup.
- `audio/`: audio uploads, voiceover generation, and transition sound effects.
- `expressions/`: expression assets, Gemini expression mapping, and overlays.
- `images/`: low-level image preprocessing and vertical composition helpers.
- `manga/`: manga PDF/session processing.
- `slides/`: image-slide planning, slide uploads, and slideshow rendering.
- `text/`: script generation, subtitle generation, and text cleanup.
- `video/`: video rendering, final assembly, and visual style presets.

The flat modules in `app/utils/*.py` are compatibility shims. New imports should use the grouped package paths, for example:

```python
from app.utils.slides.image_slides import generate_image_slides
from app.utils.video.generate_final_video import generate_final_video
from app.utils.audio.tts_generator import generate_voiceover
```
