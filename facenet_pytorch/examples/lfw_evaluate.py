from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face

import torch

from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler

from torchvision import datasets, transforms
import math

import numpy as np

import os
data_dir = r'c:\users\mrdas\lovo.jpg'

pairs_path = r'c:\users\mrdas\documents\cynapto_folder\datasets\fr_test_set\fr_test_set'



batch_size = 16

epochs = 15

workers = 0 if os.name == 'nt' else 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running on device: {}'.format(device))
mtcnn = MTCNN(

    image_size=160,

    margin=14,

    device=device,

    selection_method='center_weighted_size'

)
# Define the data loader for the input set of images

orig_img_ds = datasets.ImageFolder(data_dir, transform=None)


# overwrites class labels in dataset with path so path can be used for saving output in mtcnn batches

orig_img_ds.samples = [

    (p, p)

    for p, _ in orig_img_ds.samples

]



loader = DataLoader(

    orig_img_ds,

    num_workers=workers,

    batch_size=batch_size,

    collate_fn=training.collate_pil

)

crop_paths = []

box_probs = []



for i, (x, b_paths) in enumerate(loader):

    crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]

    mtcnn(x, save_path=crops)

    crop_paths.extend(crops)

    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
# Remove mtcnn to reduce GPU memory usage

del mtcnn

torch.cuda.empty_cache()
# create dataset and data loaders from cropped images output from MTCNN



trans = transforms.Compose([

    np.float32,

    transforms.ToTensor(),

    fixed_image_standardization

])



dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)



embed_loader = DataLoader(

    dataset,

    num_workers=workers,

    batch_size=batch_size,

    sampler=SequentialSampler(dataset)

)
# Load pretrained resnet model

resnet = InceptionResnetV1(

    classify=False,

    pretrained='vggface2'

).to(device)
classes = []

embeddings = []

resnet.eval()

with torch.no_grad():

    for xb, yb in embed_loader:

        xb = xb.to(device)

        b_embeddings = resnet(xb)

        b_embeddings = b_embeddings.to('cpu').numpy()

        classes.extend(yb.numpy())

        embeddings.extend(b_embeddings)
embeddings_dict = dict(zip(crop_paths,embeddings))



from sklearn.model_selection import KFold

from scipy import interpolate



# LFW functions taken from David Sandberg's FaceNet implementation

def distance(embeddings1, embeddings2, distance_metric=0):

    if distance_metric==0:

        # Euclidian distance

        diff = np.subtract(embeddings1, embeddings2)

        dist = np.sum(np.square(diff),1)

    elif distance_metric==1:

        # Distance based on cosine similarity

        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)

        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)

        similarity = dot / norm

        dist = np.arccos(similarity) / math.pi

    else:

        raise 'Undefined distance metric %d' % distance_metric



    return dist



def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):

    assert(embeddings1.shape[0] == embeddings2.shape[0])

    assert(embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])

    nrof_thresholds = len(thresholds)

    k_fold = KFold(n_splits=nrof_folds, shuffle=False)



    tprs = np.zeros((nrof_folds,nrof_thresholds))

    fprs = np.zeros((nrof_folds,nrof_thresholds))

    accuracy = np.zeros((nrof_folds))



    is_false_positive = []

    is_false_negative = []



    indices = np.arange(nrof_pairs)



    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        if subtract_mean:

            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)

        else:

          mean = 0.0

        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)



        # Find the best threshold for the fold

        acc_train = np.zeros((nrof_thresholds))

        for threshold_idx, threshold in enumerate(thresholds):

            _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])

        best_threshold_index = np.argmax(acc_train)

        for threshold_idx, threshold in enumerate(thresholds):

            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])

        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])



        tpr = np.mean(tprs,0)

        fpr = np.mean(fprs,0)

        is_false_positive.extend(is_fp)

        is_false_negative.extend(is_fn)



    return tpr, fpr, accuracy, is_false_positive, is_false_negative



def calculate_accuracy(threshold, dist, actual_issame):

    predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))

    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))

    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))



    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))

    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)



    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)

    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)

    acc = float(tp+tn)/dist.size

    return tpr, fpr, acc, is_fp, is_fn



def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):

    assert(embeddings1.shape[0] == embeddings2.shape[0])

    assert(embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])

    nrof_thresholds = len(thresholds)

    k_fold = KFold(n_splits=nrof_folds, shuffle=False)



    val = np.zeros(nrof_folds)

    far = np.zeros(nrof_folds)



    indices = np.arange(nrof_pairs)



    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        if subtract_mean:

            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)

        else:

          mean = 0.0

        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)



        # Find the threshold that gives FAR = far_target

        far_train = np.zeros(nrof_thresholds)

        for threshold_idx, threshold in enumerate(thresholds):

            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])

        if np.max(far_train)>=far_target:

            f = interpolate.interp1d(far_train, thresholds, kind='slinear')

            threshold = f(far_target)

        else:

            threshold = 0.0



        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])



    val_mean = np.mean(val)

    far_mean = np.mean(far)

    val_std = np.std(val)

    return val_mean, val_std, far_mean



def calculate_val_far(threshold, dist, actual_issame):

    predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))

    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    n_same = np.sum(actual_issame)

    n_diff = np.sum(np.logical_not(actual_issame))

    val = float(true_accept) / float(n_same)

    far = float(false_accept) / float(n_diff)

    return val, far







def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):

    # Calculate evaluation metrics

    thresholds = np.arange(0, 4, 0.01)

    embeddings1 = embeddings[0::2]

    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embeddings1, embeddings2,

        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    thresholds = np.arange(0, 4, 0.001)

    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,

        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    return tpr, fpr, accuracy, val, val_std, far, fp, fn



def add_extension(path):

    if os.path.exists(path+'.jpg'):

        return path+'.jpg'

    elif os.path.exists(path+'.png'):

        return path+'.png'

    else:

        raise RuntimeError('No file "%s" with extension png or jpg.' % path)



def get_paths(lfw_dir, pairs):

    nrof_skipped_pairs = 0

    path_list = []

    issame_list = []

    for pair in pairs:

        if len(pair) == 3:

            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))

            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))

            issame = True

        elif len(pair) == 4:

            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))

            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))

            issame = False

        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist

            path_list += (path0,path1)

            issame_list.append(issame)

        else:

            nrof_skipped_pairs += 1

    if nrof_skipped_pairs>0:

        print('Skipped %d image pairs' % nrof_skipped_pairs)



    return path_list, issame_list



def read_pairs(pairs_filename):

    pairs = []

    with open(pairs_filename, 'r') as f:

        for line in f.readlines()[1:]:

            pair = line.strip().split()

            pairs.append(pair)

    return np.array(pairs, dtype=object)
pairs = read_pairs(pairs_path)

path_list, issame_list = get_paths(data_dir+'_cropped', pairs)

embeddings = np.array([embeddings_dict[path] for path in path_list])



tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)
print(accuracy)

np.mean(accuracy)



