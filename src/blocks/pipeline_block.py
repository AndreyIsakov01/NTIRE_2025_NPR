from abc import ABC, abstractmethod


class BaseBlock(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class PipelineBlock(BaseBlock):
    def __init__(self, previous_block: BaseBlock):
        self.previous_block = previous_block

    def __getitem__(self, index):
        return self.previous_block[index]

    def __len__(self):
        return len(self.previous_block)