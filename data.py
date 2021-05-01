from glob import glob
from torch.utils.data import Dataset, IterableDataset, ConcatDataset, ChainDataset


def get_filenames(pathname):
    return glob(pathname + "*")


class WordDataset(Dataset):
    def __init__(self, filename):
        with open(filename) as file:
            self.sentences = file.readlines()

    def __getitem__(self, index):
        return self.sentences[index].strip().split()

    def __len__(self):
        return len(self.sentences)


class WordIterableDataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as file:
            for line in file:
                yield line.strip().split()


class OneBillionWordDataset(ConcatDataset):
    def __init__(self, pathname):
        files = get_filenames(pathname)
        super().__init__([WordDataset(file) for file in sorted(files)])


class OneBillionWordIterableDataset(ChainDataset):
    def __init__(self, pathname):
        files = get_filenames(pathname)
        super().__init__([WordIterableDataset(file) for file in sorted(files)])
