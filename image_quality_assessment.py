import imageio
from cv2 import cv2
import numpy as np
import cpbd
from scipy.stats import entropy


class Image:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self.image = cv2.imread(self.path)
        self.image_hsl = imageio.imread(self.path, pilmode="L")
        self.height, self.width = self.image.shape[:2]

    def rgb_to_relative_luminance(self, xpixel: int, ypixel: int) -> float:
        blue, green, red = self.image[xpixel, ypixel]
        relative_luminance = ((0.2126 * red) + (0.7152 * green) + (0.0722 * blue)) / 255

        # return round(relative_luminance, 4)
        return relative_luminance

    def get_luminance(self) -> str:
        dark = 0
        semidark = 0
        bright = 0

        avg_luminance = []

        for ypix in range(self.height):
            for xpix in range(self.width):
                luminance = self.rgb_to_relative_luminance(ypix, xpix)
                avg_luminance.append(luminance)
                if luminance < 0.315:
                    dark += 1
                elif 0.315 <= luminance <= 0.615:
                    semidark += 1
                else:
                    bright += 1

        # print(luminance)
        # return round((sum(avg_luminance) / len(avg_luminance)), 4)
        # return round(luminance, 4)

        if (dark > semidark) and (dark > bright):
            a = [i for i in avg_luminance if (i < 0.315)]
            return round((sum(a) / len(a)), 4)
            # return "dark"
        elif (semidark > dark) and (semidark > bright):
            a = [i for i in avg_luminance if (0.315 <= i <= 0.615)]
            return round((sum(a) / len(a)), 4)
            # return "semidark"
        else:
            # return "bright"
            a = [i for i in avg_luminance if (i > 0.615)]
            return round((sum(a) / len(a)), 4)

    def get_sharpness(self) -> str:
        sharp_value = cpbd.compute(self.image_hsl)  # [0.2 - 0.4]
        return round(sharp_value, 4)

    def get_average_information_entropy(self) -> float:
        histogram = [0] * 256

        for ypix in range(self.height):
            for xpix in range(self.width):
                blue, green, red = self.image[ypix, xpix]
                grayscale_pixel = 0.299 * red + 0.587 * green + 0.114 * blue
                histogram[int(grayscale_pixel)] = histogram[int(grayscale_pixel)] + 1

        histogram = np.asarray(histogram)
        prob_density = histogram / (self.height * self.width)

        info_entropy = entropy(prob_density)
        aux_var = 0

        for i in range(3):
            aux_var += (info_entropy ** 2) * i
        avg_infor_entropy = ((1 / 3) * aux_var) ** 0.5

        return round(avg_infor_entropy, 4)

    def get_colorfulness(self):

        blue, green, red = cv2.split(self.image.astype("float"))

        rg = np.absolute(red - green)
        yb = np.absolute(0.5 * (red + green) - blue)

        rbMean, rbStd = (np.mean(rg), np.std(rg))
        ybMean, ybStd = (np.mean(yb), np.std(yb))

        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

        return round(stdRoot + (0.3 * meanRoot), 4)
