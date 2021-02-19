import numpy as np


def str2ind(category_name, class_list):
    return [i for i in range(len(class_list)) if category_name == class_list[i]][0]


def smooth(v):
    return v
    # l = min(351, len(v)); l = l - (1-l%2)
    # if len(v) <= 3:
    #   return v
    # return savgol_filter(v, l, 1) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)


def filter_segments(segment_predict, video_names, ambilist, factor):
    ind = np.zeros(segment_predict.shape[0])

    for i in range(segment_predict.shape[0]):
        vn = video_names[int(segment_predict[i, 0])]
        for a in ambilist:
            if a[0] == vn:
                gt = range(int(round(float(a[2]) * factor)), int(round(float(a[3]) * factor)))
                pd = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(len(set(gt).union(set(pd))))

                if IoU > 0:
                    ind[i] = 1

    s = [segment_predict[i, :] for i in range(np.shape(segment_predict)[0]) if ind[i] == 0]
    return np.array(s)


def getLocMAP(predictions, threshold, annotation_path, args):
    # todo np.load needs pickle
    gt_segments = np.load(annotation_path + '/segments.npy', allow_pickle=True)
    gt_labels = np.load(annotation_path + '/labels.npy', allow_pickle=True)

    video_name = np.load(annotation_path + '/video_name.npy', allow_pickle=True)
    video_name = np.array([v.decode('utf-8') for v in video_name])

    subset = np.load(annotation_path + '/subset.npy', allow_pickle=True)
    subset = np.array([s.decode('utf-8') for s in subset])

    class_list = np.load(annotation_path + '/class_list.npy', allow_pickle=True)
    class_list = np.array([c.decode('utf-8') for c in class_list])

    duration = np.load(annotation_path + '/duration.npy', allow_pickle=True)
    ambi_list = annotation_path + '/Ambiguous_test.txt'

    # if args.feature_type == 'UNT':
    #     factor = 10.0 / 4.0
    # else:
    #     factor = 25.0 / 16.0

    factor = 10.0 / 4.0 if args.feature_type == 'UNT' else 25.0 / 16.0

    ambi_list = list(open(ambi_list, 'r'))
    ambi_list = [a.strip('\n').split(' ') for a in ambi_list]
    # todo len(ambi_list) == 4 ??

    # keep training gt_labels for plotting
    # gtltr = []
    # for i, s in enumerate(subset):
    #     if subset[i] == 'validation' and len(gt_segments[i]) > 0:
    #         gtltr.append(gt_labels[i])
    # gtlabelstr = gtltr
    gtlabelstr = [gt_labels[i] for i, s in enumerate(subset) if s == 'validation' and len(gt_segments[i]) > 0]

    # Keep only the test subset annotations
    # gts, gtl, vn, dn = [], [], [], []
    # for i, s in enumerate(subset):
    #     if subset[i] == 'test':
    #         gts.append(gt_segments[i])
    #         gtl.append(gt_labels[i])
    #         vn.append(video_name[i])
    #         dn.append(duration[i, 0])
    # gt_segments = gts
    # gt_labels = gtl
    # video_name = vn
    # duration = dn

    gt_segments = [gt_segments[i] for i, s in enumerate(subset) if s == 'test']
    gt_labels = [gt_labels[i] for i, s in enumerate(subset) if s == 'test']
    video_name = [video_name[i] for i, s in enumerate(subset) if s == 'test']
    duration = [duration[i, 0] for i, s in enumerate(subset) if s == 'test']

    # keep ground truth and predictions for instances with temporal annotations
    gts, gtl, vn, pred, dn = [], [], [], [], []
    for i, s in enumerate(gt_segments):
        if len(s):
            gts.append(gt_segments[i])
            gtl.append(gt_labels[i])
            vn.append(video_name[i])
            pred.append(predictions[i])
            dn.append(duration[i])
    gt_segments = gts
    gt_labels = gtl
    video_name = vn
    predictions = pred

    # which categories have temporal labels ?
    templabelcategories = sorted(list(set([l for gtl in gt_labels for l in gtl])))

    # the number index for those categories.
    templabelidx = []
    for t in templabelcategories:
        templabelidx.append(str2ind(t, class_list))

    # process the predictions such that classes having greater than a certain threshold are detected only
    predictions_mod = []
    c_score = []
    for p in predictions:
        pp = - p;
        [pp[:, i].sort() for i in range(np.shape(pp)[1])];
        pp = -pp
        c_s = np.mean(pp[:int(np.shape(pp)[0] / 8), :], axis=0)
        ind = c_s > 0.0
        c_score.append(c_s)
        new_pred = np.zeros((np.shape(p)[0], np.shape(p)[1]), dtype='float32')
        predictions_mod.append(p * ind)
    predictions = predictions_mod

    detection_results = []
    for i, vn in enumerate(video_name):
        detection_results.append([])
        detection_results[i].append(vn)

    ap = []
    for c in templabelidx:
        segment_predict = []
        # Get list of all predictions for class c
        for i in range(len(predictions)):
            tmp = smooth(predictions[i][:, c])
            threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp)) * 0.5
            vid_pred = np.concatenate([np.zeros(1), (tmp > threshold).astype('float32'), np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
            for j in range(len(s)):
                aggr_score = np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]
                if e[j] - s[j] >= 2:
                    segment_predict.append([i, s[j], e[j], np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]])
                    detection_results[i].append(
                        [class_list[c], s[j], e[j], np.max(tmp[s[j]:e[j]]) + 0.7 * c_score[i][c]])
        segment_predict = np.array(segment_predict)
        segment_predict = filter_segments(segment_predict, video_name, ambi_list, factor)

        # Sort the list of predictions for class c based on score
        if len(segment_predict) == 0:
            return 0
        segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

        # Create gt list
        segment_gt = [[i, gt_segments[i][j][0], gt_segments[i][j][1]] for i in range(len(gt_segments)) for j in
                      range(len(gt_segments[i])) if str2ind(gt_labels[i][j], class_list) == c]
        gtpos = len(segment_gt)

        # Compare predictions and gt
        tp, fp = [], []
        for i in range(len(segment_predict)):
            flag = 0.
            for j in range(len(segment_gt)):
                if segment_predict[i][0] == segment_gt[j][0]:
                    gt = range(int(round(segment_gt[j][1] * factor)), int(round(segment_gt[j][2] * factor)))
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p)))) / float(len(set(gt).union(set(p))))
                    if IoU >= threshold:
                        flag = 1.
                        del segment_gt[j]
                        break
            tp.append(flag)
            fp.append(1. - flag)
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        if sum(tp) == 0:
            prc = 0.
        else:
            prc = np.sum((tp_c / (fp_c + tp_c)) * tp) / gtpos
        ap.append(prc)

    return 100 * np.mean(ap)


def getDetectionMAP(predictions, annotation_path, args):
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dmap_list = []
    for iou in iou_list:
        print('Testing for IoU %f' % iou)
        dmap_list.append(getLocMAP(predictions, iou, annotation_path, args))

    return dmap_list, iou_list


if __name__ == '__main__':
    labels = np.load('Thumos14-Annotations/videoname.npy', allow_pickle=True)
    uniuqe = np.unique(labels)
    print()
