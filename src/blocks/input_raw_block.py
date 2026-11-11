import os
from pathlib import Path

import cv2

from src.blocks.pipeline_block import BaseBlock
from src.utils import fraction_from_json, json_read


class InputRawBlock(BaseBlock):
    """Reads a raw frame {i}.png and its metadata {i}.json from the inputs directory.

    Input-side logic is identical to src.blocks.image_block.InputImageBlock,
    but it does not require a ground truth directory.
    """

    def __init__(self, inputs_path: str | Path, inputs_ext: str = ".png"):
        if not isinstance(inputs_path, Path):
            inputs_path = Path(inputs_path)

        if not inputs_path.is_dir():
            raise FileNotFoundError(f"INPUT_DIR not found: {inputs_path}")

        self.inputs_path = inputs_path
        self.inputs_ext = inputs_ext

    def available_indices(self) -> list[int]:
        """Numeric frame names ({name}{ext}), sorted as numbers."""
        names = [
            os.path.splitext(fn)[0]
            for fn in os.listdir(self.inputs_path)
            if fn.endswith(self.inputs_ext)
        ]
        return sorted(int(name) for name in names)

    def __len__(self):
        return len(self.available_indices())

    def __getitem__(self, index: int):
        image_path = self.inputs_path / f"{index}{self.inputs_ext}"
        metadata_path = self.inputs_path / f"{index}.json"
        if not image_path.exists():
            raise FileNotFoundError(f"Frame not found: {image_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        metadata = json_read(metadata_path, object_hook=fraction_from_json)

        result = {"image": image, "name": str(index)}
        result.update(metadata)
        return result
