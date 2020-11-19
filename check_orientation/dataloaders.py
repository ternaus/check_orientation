import random
from pathlib import Path
from typing import Any, Dict, List

import albumentations as albu
import numpy as np
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        transform: albu.Compose,
        length: int = None,
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform

        if length is None:
            self.length = len(self.image_paths)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.image_paths)

        image_path = self.image_paths[idx]

        image = load_rgb(image_path, lib="cv2")

        # apply augmentations
        image = self.transform(image=image)["image"]

        orientation = random.randint(0, 3)
        image = np.ascontiguousarray(np.rot90(image, orientation))

        return {"image_id": image_path.stem, "features": tensor_from_rgb_image(image), "targets": orientation}
