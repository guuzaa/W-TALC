import numpy as np
import csv
import torch
import scipy.io as sio

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


def sio_test():
    test_set = sio.loadmat('test_set_meta.mat')
    test_video = test_set['test_videos'][0]
    print()


def load_data_test():
    from video_dataset import Dataset
    import options
    args = options.parser.parse_args()
    dataset = Dataset(args)
    feature, labels, _ = dataset.load_data(is_training=False)
    print()


def process(feature, length):
    if len(feature) > length:
        return random_extract(feature, length)
    else:
        return pad(feature, length)


def pad(feature, length):
    if feature.shape[0] <= length:
        return np.pad(feature, ((0, length - feature.shape[0]), (0, 0)), constant_values=0)
    else:
        return feature


def random_extract(feature, t_max):
    r = np.random.randint(len(feature) - t_max)
    return feature[r:r + t_max]


def __test01():
    load_data_test()
    print()


def __test02():
    labels = np.array([0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0.])
    conf = np.array([4.1769422e-03, 2.0157661e-02, 2.2128965e-01, 1.6265440e-03, 4.1107185e-02,
                     1.1112966e-02, 8.9212045e-02, 4.0669811e-01, 1.0764504e-02, 1.4308811e-03,
                     9.1715972e-04, 4.0571796e-04, 9.4124004e-02, 1.6790090e-02, 8.9557515e-03,
                     1.6055232e-02, 2.2399064e-02, 4.0571159e-03, 2.2122420e-03, 2.6507189e-02])
    sort_ind = np.argsort(-conf)
    test_sort_ind = np.argsort(conf)[::-1]
    tp = labels[sort_ind] == 1
    fp = labels[sort_ind] != 1
    n_pos = np.sum(labels)

    fp = np.cumsum(fp, dtype='float32')
    tp = np.cumsum(tp, dtype='float32')

    prec = tp / (fp + tp)
    target = prec / 0

    print()


def __test03():
    path = 'Thumos14reduced-I3D-JOINTFeatures.npy'
    target = np.load(path, encoding='bytes', allow_pickle=True)
    print()


if __name__ == '__main__':
    __test03()
