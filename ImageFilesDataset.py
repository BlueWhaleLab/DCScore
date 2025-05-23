'''
    This code is adopted from https://github.com/marcojira/fld
'''


from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random

class ImageFilesDataset(Dataset):
    """
    Creates torch Dataset from directory of images.
    Must be structured as dir/<class>/<img_name>.<extension> for `conditional=True`
    For `conditional=False`, will search recursively for all files that match the extension
    """

    def __init__(
        self, path, name=None, extension="png", transform=None, conditional=False, n=None
    ):
        self.path = path
        self.name = name
        self.extension = extension

        self.conditional = conditional  # If conditional, will get the class from the parent folder's name
        self.transform = transform
        self.files = []

        self.files_loaded = False  # For lazy loading of files
        self.n = n  # Maximum number of images per class
    
    # def load_files(self):
    #     for curr_path in Path(self.path).rglob(f"*.{self.extension}"):
    #         if self.conditional:
    #             self.files.append((curr_path, curr_path.parent.name))
    #         else:
    #             self.files.append((curr_path, 0))
    #     self.files_loaded = True
    
    def load_files(self):
        class_count = defaultdict(int)
        # 获取所有匹配的文件路径
        all_paths = list(Path(self.path).rglob(f"*.{self.extension}"))
        
        # 打乱路径列表
        random.shuffle(all_paths)
        
        for curr_path in all_paths:
            if self.conditional:
                class_name = curr_path.parent.name
                if self.n is None or class_count[class_name] < self.n:
                    self.files.append((curr_path, class_name))
                    class_count[class_name] += 1
            else:
                if self.n is None or class_count[0] < self.n:
                    self.files.append((curr_path, 0))
                    class_count[0] += 1
        self.files_loaded = True

    def __len__(self):
        if not self.files_loaded:
            self.load_files()

        return len(self.files)

    def __getitem__(self, idx):
        if not self.files_loaded:
            self.load_files()

        img_path, class_id = self.files[idx]
        if 'mnist' in self.name.lower() and 'color' not in self.name.lower():
            with Image.open(img_path).convert("L") as img:
                if self.transform:
                    img = self.transform(img)
                return img, class_id
        else:
            with Image.open(img_path).convert("RGB") as img:
                if self.transform:
                    img = self.transform(img)
                return img, class_id

    def get_class(self, idx):
        return self.files[idx][1]
