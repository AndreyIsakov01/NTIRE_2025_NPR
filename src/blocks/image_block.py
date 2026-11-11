import os
from pathlib import Path

import cv2
import numpy as np

from src.blocks.pipeline_block import BaseBlock
from src.utils import fraction_from_json, json_read


class ImageBlock(BaseBlock):
    def __init__(
        self,
        inputs_path: str | Path,
        targets_path: str | Path,
        inputs_ext: str = ".png",
        targets_ext: str = ".JPG",
    ):
        if not isinstance(inputs_path, Path):
            inputs_path = Path(inputs_path)
        if not isinstance(targets_path, Path):
            targets_path = Path(targets_path)

        raise_if_not_exists(inputs_path)
        raise_if_not_exists(targets_path)

        inputs_files = sorted(
            [
                os.path.splitext(filename)[0]
                for filename in os.listdir(inputs_path)
                if filename.endswith(inputs_ext)
            ]
        )
        targets_files = sorted(
            [
                os.path.splitext(filename)[0]
                for filename in os.listdir(targets_path)
                if filename.endswith(targets_ext)
            ]
        )
        if inputs_files != targets_files:
            raise ValueError(
                "File names in inputs and targets directories do not match."
            )

        self.inputs_path = inputs_path
        self.targets_path = targets_path
        self.inputs_ext = inputs_ext
        self.targets_ext = targets_ext

    def __len__(self):
        return len(list(self.targets_path.glob(f"*{self.targets_ext}")))


class InputImageBlock(ImageBlock):
    def __getitem__(self, index):
        image_dir_path = self.inputs_path
        image_ext = self.inputs_ext
        imread_flag = cv2.IMREAD_UNCHANGED

        image_path = image_dir_path / f"{index}{image_ext}"
        metadata_path = self.inputs_path / f"{index}.json"

        raise_if_not_exists(image_path)
        raise_if_not_exists(metadata_path)

        image = cv2.imread(image_path, imread_flag)
        metadata = json_read(metadata_path, object_hook=fraction_from_json)

        result = {"image": image}
        result.update(metadata)

        return result


class InputDemosaicedImageBlock(ImageBlock):
    def __getitem__(self, index):
        image_dir_path = self.inputs_path
        image_ext = self.inputs_ext
        imread_flag = cv2.IMREAD_COLOR_RGB

        image_path = image_dir_path / f"{index}{image_ext}"
        metadata_path = self.inputs_path / f"{index}.json"

        raise_if_not_exists(image_path)
        raise_if_not_exists(metadata_path)

        image = cv2.imread(image_path, imread_flag)
        metadata = json_read(metadata_path, object_hook=fraction_from_json)

        result = {"image": image}
        result.update(metadata)

        return result


class InputNumpyImageBlock(ImageBlock):
    def __getitem__(self, index):
        image_dir_path = self.inputs_path
        image_ext = self.inputs_ext

        image_path = image_dir_path / f"{index}{image_ext}"
        metadata_path = self.inputs_path / f"{index}.json"

        raise_if_not_exists(image_path)
        raise_if_not_exists(metadata_path)

        image = np.load(image_path)
        metadata = json_read(metadata_path, object_hook=fraction_from_json)

        result = {"image": image}
        result.update(metadata)

        return result

class TargetImageBlock(ImageBlock):
    def __getitem__(self, index):
        image_dir_path = self.targets_path
        image_ext = self.targets_ext
        imread_flag = cv2.IMREAD_COLOR_RGB

        image_path = image_dir_path / f"{index}{image_ext}"
        metadata_path = self.inputs_path / f"{index}.json"

        raise_if_not_exists(image_path)
        raise_if_not_exists(metadata_path)

        image = cv2.imread(image_path, imread_flag)
        metadata = json_read(metadata_path, object_hook=fraction_from_json)

        result = {"image": image}
        result.update(metadata)

        return result


class CropImagesBlock(ImageBlock):
    def __getitem__(self, index):
        input_image_dir_path = self.inputs_path
        target_image_dir_path = self.targets_path
        input_image_ext = self.inputs_ext
        target_image_ext = self.targets_ext
        input_imread_flag = cv2.IMREAD_UNCHANGED
        target_imread_flag = cv2.IMREAD_COLOR_RGB

        input_image_path = input_image_dir_path / f"{index}{input_image_ext}"
        target_image_path = target_image_dir_path / f"{index}{target_image_ext}"

        raise_if_not_exists(input_image_path)
        raise_if_not_exists(target_image_path)

        input_image = cv2.imread(input_image_path, input_imread_flag)
        target_image = cv2.imread(target_image_path, target_imread_flag)

        return {"input_image": input_image, "target_image": target_image}


class CropNumpyImagesBlock(ImageBlock):
    def __getitem__(self, index):
        input_image_dir_path = self.inputs_path
        target_image_dir_path = self.targets_path
        input_image_ext = self.inputs_ext
        target_image_ext = self.targets_ext

        input_image_path = input_image_dir_path / f"{index}{input_image_ext}"
        target_image_path = target_image_dir_path / f"{index}{target_image_ext}"

        raise_if_not_exists(input_image_path)
        raise_if_not_exists(target_image_path)

        input_image = np.load(input_image_path)
        target_image = np.load(target_image_path)

        return {"input_image": input_image, "target_image": target_image}


def raise_if_not_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File path {path} does not exist.")
