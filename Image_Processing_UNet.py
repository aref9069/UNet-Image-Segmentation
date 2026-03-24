"""
Image_Processing_UNet.py
Updated version: Robust UNet-based bar segmentation with proper resizing,
GPU support, accurate diameter extraction, and fallback to traditional method.



Author: Aref Aasi
"""

import cv2 as cv
import numpy as np
import torch

# Import UNet architecture (ensure unet_model.py is in the same directory)
from unet_model import UNet


class ImageDataUNet:
    """
    Processes a single image using either UNet segmentation or traditional edge detection.
    Extracts object diameter and positional measurements from the segmented region.
    """
    def __init__(self, raw_image, return_images=True,
                 model_path="unet_segmentation.pth", use_unet=True,
                 target_size=(256, 256)):
        self.return_images = return_images
        self.raw_image = raw_image
        self.use_unet = use_unet
        self.target_size = target_size

        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"UNet will run on: {self.device}")

        # Model loading
        self.model = None
        if use_unet and model_path:
            self.load_model(model_path)

        # Result attributes
        self.state = "No Image"
        self.object_present = False
        self.diameter_points = None
        self.avg_angle = 0.0
        self.center_offset = 0.0
        self.base_image = None
        self.binary_edges = None
        self.edges_displayed = None
        self.display_image = None
        self.segmentation_mask = None

        self.process_image()

    def load_model(self, model_path):
        """Load the trained UNet model from disk."""
        try:
            self.model = UNet(n_channels=1, n_classes=1, bilinear=True)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"UNet model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load UNet model ({e}). Falling back to traditional method.")
            self.use_unet = False
            self.model = None

    def preprocess_for_unet(self, image):
        """Resize and normalize a grayscale image for UNet input."""
        resized = cv.resize(image, self.target_size, interpolation=cv.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)

    def unet_segment(self, image):
        """Run UNet inference and return a binary mask resized to the original image dimensions."""
        if self.model is None:
            return None

        input_tensor = self.preprocess_for_unet(image)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output)
            mask = (prob > 0.5).float()

        mask_np = mask.squeeze().cpu().numpy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_resized = cv.resize(mask_uint8, (image.shape[1], image.shape[0]),
                                 interpolation=cv.INTER_NEAREST)
        return mask_resized

    def trim_image(self, base_image):
        """Crop the image to the region containing the object of interest."""
        _, thresh = cv.threshold(base_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        row_sums = np.sum(thresh, axis=1)
        col_sums = np.sum(thresh, axis=0)

        rows = np.where(row_sums > row_sums.max() * 0.2)[0]
        cols = np.where(col_sums > col_sums.max() * 0.9)[0]

        if len(rows) == 0 or len(cols) == 0:
            return 0, base_image.shape[0], 0, base_image.shape[1], base_image

        start_row = max(rows[0] - 50, 0)
        stop_row = min(rows[-1] + 50, base_image.shape[0])
        start_col = max(cols[0] - 20, 0)
        stop_col = min(cols[-1] + 20, base_image.shape[1])

        trimmed = base_image[start_row:stop_row, start_col:stop_col]
        return start_row, stop_row, start_col, stop_col, trimmed

    def get_edges_traditional(self, image):
        """Detect edges using Sobel filtering and Otsu thresholding."""
        sobel = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
        sobel_abs = np.absolute(sobel)
        sobel_8u = np.uint8(sobel_abs)
        _, edges = cv.threshold(sobel_8u, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return edges

    def get_top_bottom_edges_from_mask(self, mask):
        """
        Extract the top and bottom edge positions from a binary segmentation mask.
        More accurate and stable than applying Sobel to the mask directly.
        """
        edges = np.zeros_like(mask)
        height, width = mask.shape

        top_edge = np.full(width, height, dtype=np.int32)
        bottom_edge = np.full(width, -1, dtype=np.int32)

        for x in range(width):
            column = mask[:, x]
            ys = np.where(column > 0)[0]
            if len(ys) > 0:
                top_edge[x] = ys[0]
                bottom_edge[x] = ys[-1]

        valid = bottom_edge > top_edge
        edges[top_edge[valid], np.where(valid)[0]] = 255
        edges[bottom_edge[valid], np.where(valid)[0]] = 255

        return edges, top_edge, bottom_edge

    def extract_diameter_points(self, top_edge_y, bottom_edge_y, valid_cols, trim_offsets):
        """
        Compute per-column diameters and the vertical center offset in full image coordinates.
        Applies iterative outlier removal for robustness.
        """
        start_row, _, start_col, _, _ = trim_offsets

        diameters = bottom_edge_y[valid_cols] - top_edge_y[valid_cols]
        centers = top_edge_y[valid_cols] + diameters // 2 + start_row

        if len(diameters) < 10:
            return None, 0.0

        # Iterative outlier removal
        for threshold in [50, 25, 10]:
            mean_dia = np.mean(diameters)
            diameters = diameters[
                (diameters > mean_dia - threshold) & (diameters < mean_dia + threshold)
            ]

        center_line = self.base_image.shape[0] // 2
        center_offset = center_line - np.mean(centers)

        return diameters, center_offset

    def process_image(self):
        """Main image processing pipeline."""
        if self.raw_image is None:
            self.state = "No Image"
            return

        # Convert to grayscale if needed
        if len(self.raw_image.shape) == 3:
            self.base_image = cv.cvtColor(self.raw_image, cv.COLOR_BGR2GRAY)
        else:
            self.base_image = self.raw_image.copy()

        # Basic presence check
        if np.mean(self.base_image) < 15:
            self.object_present = False
            self.state = "No Object"
            self.display_image = self.base_image
            return

        # Trim to object region
        trim_offsets = self.trim_image(self.base_image)
        start_row, stop_row, start_col, stop_col, trimmed = trim_offsets

        # UNet segmentation path
        if self.use_unet and self.model is not None:
            full_mask = self.unet_segment(self.base_image)

            if full_mask is not None:
                mask_cropped = full_mask[start_row:stop_row, start_col:stop_col]

                # Morphological cleanup
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
                mask_clean = cv.morphologyEx(mask_cropped, cv.MORPH_CLOSE, kernel)
                mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_OPEN, kernel)

                edges_cropped, top_y, bottom_y = self.get_top_bottom_edges_from_mask(mask_clean)

                if self.return_images:
                    self.segmentation_mask = full_mask
            else:
                edges_cropped, top_y, bottom_y = None, None, None
        else:
            edges_cropped = self.get_edges_traditional(trimmed)
            top_y, bottom_y = None, None

        # Fallback to traditional edge detection if UNet output is insufficient
        if edges_cropped is None or np.count_nonzero(edges_cropped) < 50:
            edges_cropped = self.get_edges_traditional(trimmed)

        # Build full-size edge images
        if self.return_images:
            self.binary_edges = np.zeros_like(self.base_image)
            self.binary_edges[start_row:stop_row, start_col:stop_col] = edges_cropped

            edges_colored = cv.cvtColor(edges_cropped, cv.COLOR_GRAY2BGR)
            edges_colored[edges_colored > 0] = [0, 255, 0]
            overlay = cv.cvtColor(self.base_image, cv.COLOR_GRAY2BGR)
            overlay[start_row:stop_row, start_col:stop_col] = cv.addWeighted(
                overlay[start_row:stop_row, start_col:stop_col], 0.7, edges_colored, 0.3, 0
            )
            self.edges_displayed = overlay

        # Extract diameter measurements
        if top_y is not None and bottom_y is not None:
            valid_cols = np.where(bottom_y > top_y)[0]
            self.diameter_points, self.center_offset = self.extract_diameter_points(
                top_y, bottom_y, valid_cols, trim_offsets
            )
        else:
            # Fallback: derive top/bottom from edge pixel positions
            min_y = np.where(
                edges_cropped > 0,
                np.arange(edges_cropped.shape[0])[:, None],
                edges_cropped.shape[0]
            ).min(axis=0)
            max_y = np.argmax(edges_cropped, axis=0)
            valid = max_y < min_y
            self.diameter_points, self.center_offset = self.extract_diameter_points(
                max_y, min_y, valid, trim_offsets
            )

        # Set final state
        if self.diameter_points is not None and len(self.diameter_points) > 10:
            self.object_present = True
            self.state = "Edges Found"
            self.avg_angle = 0.0

            self.display_image = self.edges_displayed if self.return_images else self.base_image
        else:
            self.object_present = False
            self.state = "No Edges"
            self.display_image = self.base_image


if __name__ == "__main__":
    import glob

    image_paths = glob.glob("test_images/*.jpg")
    print(f"Found {len(image_paths)} test images")

    model_path = "unet_segmentation.pth"

    for i, path in enumerate(image_paths[:5]):
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        processor = ImageDataUNet(img, return_images=True,
                                  model_path=model_path, use_unet=True)

        n_diameters = len(processor.diameter_points) if processor.diameter_points is not None else 0
        print(f"Image {i + 1}: state={processor.state}, diameter_points={n_diameters}")

        if processor.edges_displayed is not None:
            cv.imwrite(f"result_unet_{i + 1}.jpg", processor.edges_displayed)