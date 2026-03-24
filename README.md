# UNet Image Segmentation

A PyTorch implementation of the UNet architecture for binary image segmentation, with an end-to-end processing pipeline that includes preprocessing, inference, morphological cleanup, edge extraction, and diameter measurement. Falls back gracefully to traditional Sobel edge detection when a trained model is unavailable.

---

## Repository Structure

```
.
├── unet_model.py              # UNet architecture (encoder, decoder, skip connections)
├── image_processing_unet.py   # Full inference and measurement pipeline
├── unet_segmentation.pth      # Trained model weights (not included — see below)
├── test_images/               # Sample input images (You should include your dataset)
└── README.md
```

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenCV (`cv2`)
- NumPy

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy
```

GPU inference is used automatically if CUDA is available; otherwise the model runs on CPU.

---

## Installation

```bash
git clone https://github.com/your-username/unet-segmentation.git
cd unet-segmentation
pip install torch torchvision opencv-python numpy
```

---

## Model Weights

Trained weights are not included in this repository. You can either:

- **Train your own model** using the `UNet` class in `unet_model.py`
- **Use a pre-trained checkpoint** — place it in the project root and pass the path via `model_path`

The pipeline will automatically fall back to traditional Sobel edge detection if no model file is found.

---

## Quick Start

```python
import cv2 as cv
from image_processing_unet import ImageDataUNet

# Load a grayscale image
img = cv.imread("test_images/sample.jpg", cv.IMREAD_GRAYSCALE)

# Run the pipeline
result = ImageDataUNet(
    raw_image=img,
    return_images=True,
    model_path="unet_segmentation.pth",
    use_unet=True,
    target_size=(256, 256)
)

print(f"State:          {result.state}")
print(f"Object found:   {result.object_present}")
print(f"Center offset:  {result.center_offset:.2f} px")

if result.diameter_points is not None:
    print(f"Mean diameter:  {result.diameter_points.mean():.2f} px")

# Save the annotated output
if result.edges_displayed is not None:
    cv.imwrite("output.jpg", result.edges_displayed)
```

---

## `ImageDataUNet` — Parameters & Attributes

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_image` | `np.ndarray` | — | Input image (grayscale or BGR) |
| `return_images` | `bool` | `True` | Whether to produce annotated image outputs |
| `model_path` | `str` | `"unet_segmentation.pth"` | Path to trained model weights |
| `use_unet` | `bool` | `True` | Use UNet inference; falls back to Sobel if `False` or model fails to load |
| `target_size` | `tuple` | `(256, 256)` | Resize resolution for UNet input |

### Output Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `state` | `str` | `"Edges Found"`, `"No Edges"`, `"No Object"`, or `"No Image"` |
| `object_present` | `bool` | Whether a segmentable object was detected |
| `diameter_points` | `np.ndarray` \| `None` | Per-column diameter measurements (pixels) |
| `center_offset` | `float` | Vertical offset of object center from image midline (pixels) |
| `avg_angle` | `float` | Estimated tilt angle (reserved for future use) |
| `display_image` | `np.ndarray` | Annotated output image |
| `binary_edges` | `np.ndarray` | Full-size binary edge map |
| `segmentation_mask` | `np.ndarray` \| `None` | Raw UNet output mask (debug) |

---

## UNet Architecture

The model follows the standard UNet encoder–decoder structure with skip connections.

```
Input (1×H×W)
    │
    ├─ DoubleConv → 64          ──────────────────────────────┐
    ├─ Down → 128               ─────────────────────────┐    │
    ├─ Down → 256               ────────────────────┐    │    │
    ├─ Down → 512               ───────────────┐    │    │    │
    └─ Down → 512 (bottleneck)                 │    │    │    │
                                               │    │    │    │
    ┌─ Up ← 512 + 512 ──────────────────────────┘    │    │    │
    ├─ Up ← 256 + 256 ───────────────────────────────┘    │    │
    ├─ Up ← 128 + 128 ────────────────────────────────────┘    │
    └─ Up ←  64 +  64 ─────────────────────────────────────────┘
         │
    OutConv (1×1) → n_classes
```

### `UNet` Constructor Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `n_channels` | `1` | Input channels (1 = grayscale, 3 = RGB) |
| `n_classes` | `1` | Output segmentation classes |
| `bilinear` | `False` | `True` = bilinear upsampling; `False` = transposed convolution |

---

## Batch Testing

```python
import glob
import cv2 as cv
from image_processing_unet import ImageDataUNet

image_paths = glob.glob("test_images/*.jpg")

for i, path in enumerate(image_paths):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    result = ImageDataUNet(img, model_path="unet_segmentation.pth")

    n = len(result.diameter_points) if result.diameter_points is not None else 0
    print(f"[{i+1}] {path}  →  state={result.state}, measurements={n}")

    if result.edges_displayed is not None:
        cv.imwrite(f"result_{i+1}.jpg", result.edges_displayed)
```

---

## Processing Pipeline

1. **Grayscale conversion** — converts BGR input if needed
2. **Presence check** — skips dark/empty frames (mean pixel < 15)
3. **ROI crop** — trims to the detected object region to reduce noise
4. **UNet inference** — resizes to `target_size`, runs forward pass, resizes mask back
5. **Morphological cleanup** — closing then opening with a 5×5 elliptical kernel
6. **Edge extraction** — top/bottom foreground pixel per column from the binary mask
7. **Diameter calculation** — per-column height with iterative outlier removal (±50 → ±25 → ±10 px)
8. **Fallback** — if UNet output is absent or sparse (<50 non-zero pixels), Sobel edge detection is used instead

---

## License

MIT License. See [LICENSE](LICENSE) for details.
