import imageio
from concurrent import futures

import cpbd
import numpy as np
from cv2 import cv2
from scipy.stats import entropy


class Image:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self.image = cv2.imread(self.path)
        self.image_gray = cv2.imread(self.path, 0)
        self.image_xyz = cv2.cvtColor(self.image, cv2.COLOR_BGR2XYZ)
        self.image_hsl = imageio.imread(self.path, pilmode="L")
        self.height, self.width = self.image.shape[:2]

    def get_sharpness(self) -> str:
        # sharp_value = cpbd.compute(self.image_hsl)  # [0.2 - 0.4]
        from compute_cpbd import compute

        sharp_value = compute(self.image_hsl)
        return round(sharp_value, 4)

    def get_colorfulness(self):

        blue, green, red = cv2.split(self.image.astype("float"))
        rg = np.absolute(red - green)
        yb = np.absolute(0.5 * (red + green) - blue)
        rbMean, rbStd = (np.mean(rg), np.std(rg))
        ybMean, ybStd = (np.mean(yb), np.std(yb))
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

        return round(stdRoot + (0.3 * meanRoot), 4)

    def get_avg_luminance(self):
        _, y, _ = cv2.split(self.image_xyz)
        y = y / 255
        dark = y[y < 0.135]
        normal = y[(y >= 0.135) & (y <= 0.615)]
        brihgt = y[y > 0.615]

        if dark.size > normal.size and dark.size > brihgt.size:
            luminance = sum(dark) / len(dark)
        elif normal.size > dark.size and normal.size > brihgt.size:
            luminance = sum(normal) / len(normal)
        else:
            luminance = sum(brihgt) / len(brihgt)

        return round(luminance, 4)

    def get_avg_information(self):
        histogram = cv2.calcHist([self.image_gray], [0], None, [256], [0, 256])
        prob_density = histogram / (self.height * self.width)
        info_entropy = float(entropy(prob_density))
        aux_var = 0

        for i in range(3):
            aux_var += (info_entropy ** 2) * i
        avg_infor_entropy = ((1 / 3) * aux_var) ** 0.5

        return round(avg_infor_entropy, 4)

    def get_features(self):
        features = [
            self.get_sharpness(),
            self.get_avg_luminance(),
            self.get_avg_information(),
            self.get_colorfulness(),
        ]

        return features
