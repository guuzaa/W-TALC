import numpy as np
import csv

from detectionMAP import str2ind


def __test():
    video_name = np.load('Thumos14-Annotations/videoname.npy', allow_pickle=True)
    video_name = np.array([v.decode('utf-8') for v in video_name])

    gt_segments = np.load('Thumos14-Annotations/segments.npy', allow_pickle=True)

    gt_labels = np.load('Thumos14-Annotations/labels.npy', allow_pickle=True)

    subset = np.load('Thumos14-Annotations/subset.npy', allow_pickle=True)
    subset = np.array([s.decode('utf-8') for s in subset])

    class_list = np.load('Thumos14-Annotations/classlist.npy', allow_pickle=True)
    class_list = np.array([c.decode('utf-8') for c in class_list])

    duration = np.load('Thumos14-Annotations/duration.npy', allow_pickle=True)

    gt_label_str = [gt_labels[i] for i, s in enumerate(subset) if s == 'validation' and len(gt_segments) > 0]
    gt_segments = [gt_segments[i] for i, s in enumerate(subset) if s == 'test']
    gt_labels = [gt_labels[i] for i, s in enumerate(subset) if s == 'test']
    video_name = [video_name[i] for i, s in enumerate(subset) if s == 'test']
    duration = [duration[i, 0] for i, s in enumerate(subset) if s == 'test']

    gt_labels = [gt_labels[i] for i, s in enumerate(gt_segments) if len(s) > 0]
    video_name = [video_name[i] for i, s in enumerate(gt_segments) if len(s) > 0]
    # predictions = [predictions[i] for i, s in enumerate(gt_segments) if len(s) > 0]
    gt_segments = [gt_segments[i] for i, s in enumerate(gt_segments) if len(s) > 0]

    temp_label_categories = sorted(list(set(l for gtl in gt_labels for l in gtl)))

    temporal_label_idx = [str2ind(t, class_list) for t in temp_label_categories]

    detection_results = []
    for i, vn in enumerate(video_name):
        detection_results.append([vn])

    test_detection_results = [[vn] for vn in video_name]

    target = (detection_results == test_detection_results)




    print()


if __name__ == '__main__':
    __test()
