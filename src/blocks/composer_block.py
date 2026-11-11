from src.blocks.pipeline_block import PipelineBlock


class ComposerBlock(PipelineBlock):
    def __init__(self, input_ds, target_ds):
        if len(input_ds) != len(target_ds):
            raise ValueError("Input and target datasets must have the same length")

        self.input_ds = input_ds
        self.target_ds = target_ds

    def __getitem__(self, index):
        return {
            "input_image": self.input_ds[index]["image"],
            "target_image": self.target_ds[index]["image"],
            "target_image_mask": self.target_ds[index]["image_mask"],
        }

    def __len__(self):
        return len(self.target_ds)
