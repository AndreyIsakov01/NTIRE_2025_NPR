import numpy as np

from src.blocks import PipelineBlock


class FlashToneMappingBlock(PipelineBlock):
    def __init__(self, previous_block: PipelineBlock, scaling_parameter: float = 5):
        super().__init__(previous_block)

        self.scaling_parameter = scaling_parameter

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        values = np.max(image, axis=2)
        values_copy = np.copy(values)
        values_copy[values_copy == 0] = 1e-9
        mapped_image = image / (
            self.scaling_parameter * np.exp(np.mean(np.log(values_copy)))
            + np.tile(np.expand_dims(values_copy, axis=2), (1, 1, 3))
        )

        mapped_image *= 1 / np.max(mapped_image)

        previous_block_result["image"] = mapped_image

        return previous_block_result


class UnrealToneMappingBlock(PipelineBlock):
    def __init__(self, previous_block: PipelineBlock, slope: float, offset: float):
        super().__init__(previous_block)

        self.slope = slope
        self.offset = offset

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        mapped_image = image / (image + self.offset) * self.slope
        # mapped_image = np.clip(mapped_image, 0.0, 1.0)

        previous_block_result["image"] = mapped_image

        return previous_block_result


class SigmoidToneMappingBlock(PipelineBlock):
    def __init__(self, previous_block: PipelineBlock, slope: float, offset: float):
        super().__init__(previous_block)

        self.slope = slope
        self.offset = offset

    def __getitem__(self, index):
        previous_block_result = self.previous_block[index]

        image = previous_block_result["image"]

        mapped_image = 1 / (1 + np.exp(-self.slope * (image - self.offset)))
        mapped_image = np.clip(mapped_image, 0.0, 1.0)

        previous_block_result["image"] = mapped_image

        return previous_block_result
