import numpy as np
import csv


def __test():
    video_name = np.load('Thumos14-Annotations/videoname.npy', allow_pickle=True)
    video_name = np.array([v.decode('utf-8') for v in video_name])

    gt_segments = np.load('Thumos14-Annotations/segments.npy', allow_pickle=True)

    subset = np.load('Thumos14-Annotations/subset.npy', allow_pickle=True)
    subset = np.array([s.decode('utf-8') for s in subset])

    duration = np.load('Thumos14-Annotations/duration.npy', allow_pickle=True)

    print()


if __name__ == '__main__':
    __test()
