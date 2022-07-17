from torch.utils.data.dataloader import DataLoader


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
