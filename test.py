import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

import utils
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP import getDetectionMAP as dmAP

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def test(itr, dataset, args, model, logger, device, verbose=False):
    done = False
    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []

    while not done:
        if dataset.currenttestidx % 100 == 0 and verbose:
            print('Testing test data point %d of %d' % (dataset.currenttestidx, len(dataset.testidx)))

        features, labels, done = dataset.load_data(is_training=False)
        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            _, element_logits = model(features, is_training=False)

        instance_logits = F.softmax(
            torch.mean(torch.topk(element_logits, k=int(np.ceil(len(features) / 8)), dim=0)[0], dim=0),
            dim=0).cpu().data.numpy()
        element_logits = element_logits.cpu().data.numpy()

        instance_logits_stack.append(instance_logits)
        element_logits_stack.append(element_logits)
        labels_stack.append(labels)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)

    dmap, ious = dmAP(element_logits_stack, dataset.path_to_annotations, args)

    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                labels_stack[i, :] = np.zeros_like(labels_stack[i, :])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('-' * 16, 'result', '-' * 16)
    print('Classification map %f' % cmap)

    for i, j in zip(ious, dmap):
        print('Detection map @ %.2f = %.2f' % (i, j))

    print('-' * 40)

    logger.log_value('Test Classification mAP', cmap, itr)
    for item in list(zip(dmap, ious)):
        logger.log_value('Test Detection mAP @ IoU = ' + str(item[1]), item[0], itr)

    utils.write_to_file(args.dataset_name, dmap, cmap, itr)
