import numpy as np


def _getAP(conf, labels):
    assert len(conf) == len(labels)
    sort_ind = np.argsort(-conf)
    tp = labels[sort_ind] == 1
    fp = labels[sort_ind] != 1
    n_pos = np.sum(labels)

    # todo why np.cumsum
    fp = np.cumsum(fp, dtype='float32')
    tp = np.cumsum(tp, dtype='float32')

    # recall = tp / n_pos
    prec = tp / (fp + tp)

    tmp = (labels[sort_ind] == 1).astype('float32')

    return np.sum(tmp * prec) / n_pos


def getClassificationMAP(confidence, labels):
    """ confidence and labels are of dimension n_samples x n_label """

    # print(labels.shape)
    # AP = []
    # for i in range(np.shape(labels)[1]):
    #     AP.append(_getAP(confidence[:, i], labels[:, i]))

    AP = [_getAP(confidence[i], labels[i]) for i in range(labels.shape[0])]

    return 100 * np.nansum(AP) / len(AP)
