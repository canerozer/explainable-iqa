r"""
A Basic Implementation of the Pointing Game

Some parts of these implementation were obtained from
https://github.com/facebookresearch/TorchRay/blob/master/torchray/benchmark/pointing_game.py
"""

import random

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max


class PointingGame:
    def __init__(self, num_classes, save_to_file=False, path=None,
                 tolerance=15, mode="singlepeak", pc=95,
                 min_distance=100):
        self.num_classes = num_classes
        self.tolerance = tolerance

        self.save_to_file = save_to_file
        if save_to_file:
            assert path != None
            self.data = []
            self.path = path

        self.mode = mode
        self.pc = pc
        self.min_distance = min_distance

        self.hits_any = np.zeros(num_classes)
        self.misses_any = np.zeros(num_classes)
        self.hits_all = np.zeros(num_classes)
        self.misses_all = np.zeros(num_classes)
        self.all_scores = []

    def generate_binary_mask(self, boxes, shape):
        mask = np.zeros(shape)
        for box in boxes:
            box = list(map(int, box))
            xmin, ymin, xmax, ymax = box
            mask[xmin:xmax, ymin:ymax] = 1
        return mask.astype(bool)

    def obtain_peaks(self, map):
        if self.mode == "singlepeak":
            peaks = self.obtain_max_point(map)
            peaks = [peaks[0]]
        elif self.mode == "randompeak":
            peaks = self.obtain_max_point(map)
            peaks = [random.choice(peaks)]
        if self.mode == "average_score_max_peak":
            peaks = self.obtain_max_point(map)
        if self.mode == "average_score_top_peaks":
            peaks = self.obtain_max_points(map)
        elif self.mode == "maxpeaks":
            peaks = self.obtain_max_point(map)
        elif self.mode == "multipeak":
            peaks = self.obtain_max_points(map)
        elif self.mode == "multipeak_local":
            peaks = self.obtain_peak_local_points(map)
        return peaks

    def obtain_max_point(self, map):
        max_val = np.amax(map)
        loc = np.where(map == max_val)
        listOfCordinates = list(zip(loc[0], loc[1]))
        return listOfCordinates

    def obtain_max_points(self, map):
        thr = np.percentile(map, self.pc)
        loc = np.where(map > thr)
        listOfCordinates = list(zip(loc[0], loc[1]))
        return listOfCordinates

    def obtain_peak_local_points(self, map):
        loc = peak_local_max(map, min_distance=self.min_distance)
        listOfCordinates = list(zip(loc[:, 0], loc[:, 1]))
        return listOfCordinates

    def evaluate(self, map, boxes=None, mask=None, name=None):
        # Check the ndim of the map
        assert len(map.shape) == 2

        # Use boxes when there are no mask
        if mask is None:
            mask = self.generate_binary_mask(boxes, map.shape)

        if np.all(mask == False):
            return 0, 0

        points = self.obtain_peaks(map)

        hits = []

        for point in points:
            v, u = np.meshgrid(
                (np.arange(mask.shape[0], dtype=np.float) - point[0])**2,
                (np.arange(mask.shape[1], dtype=np.float) - point[1])**2,
            )
            accept = (v + u) < self.tolerance**2

            hit = (mask & accept).any()

            hits.append(hit)

        output_any = +1 if np.any(hits) else -1
        output_all = +1 if np.all(hits) else -1

        if self.save_to_file:
            self.record(name, output_any, output_all)

        if self.mode in ["average_score_max_peak", "average_score_top_peaks"]:
            hits_float = np.array(hits).astype(np.float)
            if len(hits_float) == 0:
                raise IndexError("Hits list is empty")
            return hits_float.mean(), None
        else:
            return output_any, output_all

    def accumulate(self, hit, class_id, target="any"):
        if target == "any":
            if hit == 0:
                return
            if hit == 1:
                self.hits_any[class_id] += 1
            elif hit == -1:
                self.misses_any[class_id] += 1
            else:
                assert False
        elif target == "all":
            if hit == 0:
                return
            if hit == 1:
                self.hits_all[class_id] += 1
            elif hit == -1:
                self.misses_all[class_id] += 1
            else:
                assert False
        elif target == "average_score":
            self.all_scores.append(hit)

    def print_stats(self):
        if self.mode in ["average_score_max_peak", "average_score_top_peaks"]:
            mean_score = np.mean(self.all_scores)
            print(f"Average score: {mean_score}")
        else:
            print(f"Hits Any: {self.hits_any}")
            print(f"Misses Any: {self.misses_any}")
            self.acc_any = self.hits_any / (self.hits_any + self.misses_any)
            print(f"Accuracy Any: {self.acc_any}")

            print(f"Hits All: {self.hits_all}")
            print(f"Misses All: {self.misses_all}")
            self.acc_all = self.hits_all / (self.hits_all + self.misses_all)
            print(f"Accuracy All: {self.acc_all}")

    def record(self, name, output_any, output_all):
        self.data.append((name, output_any, output_all))

    def emit(self):
        self.data.append(("Total", self.acc_any, self.acc_all))
        df = pd.DataFrame(self.data, columns=["Filename", "Hit/Miss (Any)",
                                              "Hit/Miss (All)"])
        if self.mode == "multipeak_local":
            df.to_csv(self.path + f"_d={self.min_distance}" + "_stats.csv")
        else:
            df.to_csv(self.path + "_stats.csv")

