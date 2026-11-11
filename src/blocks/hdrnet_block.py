import cv2
import numpy as np
import torch

from src.blocks.pipeline_block import PipelineBlock


class HDRNetBlock(PipelineBlock):
    """Replaces the tone-mapping block with a trained HDRNet (``HDRPointwiseNN``).

    Expects ``image`` as a float32 RGB array in [0, 1] on input (e.g., the
    output of ``SRGBTransformBlock``). The low-resolution input is built with
    ``INTER_NEAREST`` -- the same way as in ``HDRContrastDataset``, after which
    the model is applied to each tile.

    If ``tile_size`` is set and the image is larger than a tile, the image is
    cut into overlapping tiles (the tile size matches the training
    ``net_output_size``), and the result is stitched together with linear
    feathering on the overlaps -- this guarantees bounded memory and a scale
    consistent with training.
    """

    def __init__(
        self,
        previous_block: PipelineBlock,
        model: torch.nn.Module,
        device: torch.device,
        lowres_size: int = 256,
        tile_size: int | None = None,
        tile_overlap: float = 0.25,
    ):
        super().__init__(previous_block)
        self.model = model
        self.device = device
        self.lowres_size = lowres_size
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("HDRNetBlock expects an HxWx3 image")
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        h, w = image.shape[:2]
        tile = self.tile_size
        if tile and (h > tile or w > tile):
            out = self._run_tiled(image, tile)
        else:
            out = self._run_model(image)

        previous_block_result["image"] = out.astype(np.float32)
        return previous_block_result

    def _to_device(self, arr):
        return (
            torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

    def _run_model(self, full):
        low = cv2.resize(
            full,
            (self.lowres_size, self.lowres_size),
            interpolation=cv2.INTER_NEAREST,
        )

        low_t = self._to_device(low)
        full_t = self._to_device(full)

        with torch.inference_mode():
            self.model.eval()
            pred = self.model(low_t, full_t)

        pred_np = (
            pred.squeeze(0)
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )

        del low_t, full_t, pred
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pred_np

    @staticmethod
    def _tile_starts(length, tile, overlap):
        if tile >= length:
            return [0]
        if overlap <= 0:
            starts = list(range(0, length - tile + 1, tile))
            if starts[-1] + tile < length:
                starts.append(length - tile)
            return starts
        step = max(1, int(tile * (1.0 - overlap)))
        starts = list(range(0, length - tile + 1, step))
        if starts[-1] + tile < length:
            starts.append(length - tile)
        return starts

    def _run_tiled(self, image, tile):
        h, w = image.shape[:2]
        overlap = self.tile_overlap

        acc = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w, 1), np.float32)

        ys = self._tile_starts(h, tile, overlap)
        xs = self._tile_starts(w, tile, overlap)

        for y0 in ys:
            for x0 in xs:
                pred = self._run_model(image[y0 : y0 + tile, x0 : x0 + tile])

                wy = np.ones((pred.shape[0], 1), np.float32)
                wx = np.ones((pred.shape[1], 1), np.float32)

                ramp_h = int(tile * overlap) + 1
                if y0 > 0:
                    n = min(y0 + 1, ramp_h)
                    wy[:n, 0] = np.linspace(0.0, 1.0, n)
                if y0 + tile < h:
                    n = min(h - (y0 + tile), ramp_h)
                    wy[-n:, 0] = np.linspace(1.0, 0.0, n)

                ramp_w = int(tile * overlap) + 1
                if x0 > 0:
                    n = min(x0 + 1, ramp_w)
                    wx[:n, 0] = np.linspace(0.0, 1.0, n)
                if x0 + tile < w:
                    n = min(w - (x0 + tile), ramp_w)
                    wx[-n:, 0] = np.linspace(1.0, 0.0, n)

                weight = (wy @ wx.T)[..., None]
                acc[y0 : y0 + tile, x0 : x0 + tile] += pred * weight
                wsum[y0 : y0 + tile, x0 : x0 + tile] += weight

        return acc / np.maximum(wsum, 1e-6)