from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os
import onnx
import onnxruntime as ort
import cv2
from PIL import Image
import numpy as np
import os
import glob

lfw_dir = r"c:\users\mrdas\documents\cynapto_folder\datasets\fr_resized_cropped"
pairs_filename = r"c:\users\mrdas\documents"
# Recursively search for image files in the directory and its subdirectories
img_paths = glob.glob(os.path.join(lfw_dir, '**/*.jpg'), recursive=True)

#img_paths=get_paths(lfw_dir,pairs_filename)
# Resize and normalize each image
imgs = []
imgs_path = []
for img_path in img_paths:
    # Load image from disk
    imgs_path.append(img_path)
    img = Image.open(img_path)

    # Resize image to 160x160 pixels
    img = img.resize((160, 160), resample=Image.BILINEAR)

    # Convert image to numpy array and normalize pixel values
    img = np.array(img)
    img = (img - 127.5) / 128.0

    # Add image to list of images
    imgs.append(img)

# Convert list of images to numpy array
imgs = np.array(imgs)

# Convert numpy array to PyTorch tensor
imgs_tensor = torch.from_numpy(imgs)
print(imgs_tensor[-1])
print(imgs_path[-1])

pairs_filename = r"C:\Users\kaupk\Images\pairs1.txt"
lfw_dir = r"C:\Users\kaupk\Images\lfw-deepfunneled"
batch_size = 16
#session = onnxruntime.InferenceSession(r"C:\Users\kaupk\Downloads\webface_r50.onnx")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
model_path = r"C:\Users\kaupk\Downloads\webface_r50.onnx"
ort_session = ort.InferenceSession(model_path)

embs = []
for img in imgs:
    img = cv2.resize(img, (112, 112))
    #img = prewhiten(img)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...].astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outputs = ort_session.run(None, ort_inputs)
    emb = ort_outputs[0][0]
    embs.append(emb)
embs = np.stack(embs)
print(embs[-1])
#print(embs.shape)

embeddings_dict = dict(zip(imgs_path,embs))
#print(embeddings_dict)

from sklearn.model_selection import KFold
from scipy import interpolate
import math

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
    val=0.0
    far=0.0
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    try:
        val = float(true_accept) / float(n_same)
    except:
        pass
    try:
        far = float(false_accept) / float(n_diff)
    except:pass
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10,distance_metric=0, subtract_mean=False):

    # Calculate evaluation metrics
    thresholds = ()
    a=np.array(embs)
    embeddings1 = a[0::2]
    embeddings2 = a[1::2]
    thresholds = np.arange(0, 4, 0.001)

    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
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
    path0 = 0
    path1 = 0
    issame = False

    for pair in pairs:
        a=pair[0]
        b=a.split(",")
      
        if len(b) == 3:
            try:
                path0 = add_extension(os.path.join(lfw_dir, b[0], b[0] + '_' + '%04d' % int(b[1])))
                path1 = add_extension(os.path.join(lfw_dir, b[0], b[0] + '_' + '%04d' % int(b[2])))
                #print(path0)
                issame = True
            except:
                pass
        elif len(b) == 4:
            try:
                path0 = add_extension(os.path.join(lfw_dir, b[0], b[0] + '_' + '%04d' % int(b[1])))
                path1 = add_extension(os.path.join(lfw_dir, b[2], b[2] + '_' + '%04d' % int(b[3])))
                issame = False
            except:
                pass
        if os.path.exists(path0) and os.path.exists(path1):  
            #issame = True  # Only add the pair if both paths exist
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
            pair = line.strip().split(",")
            pairs.append(pair)
    return np.array(pairs, dtype=object)

pairs = read_pairs(pairs_filename)
print(pairs[:6])
crop_dir = r"C:\Users\kaupk\Images\lfw-deepfunneled"
path_list, issame_list = get_paths(crop_dir, pairs)
#print(path_list)
#b=np.array(path_list)

#path_list, issame_list = get_paths(data_dir+'_cropped', pairs)
#print(path_list)
#embeddings = np.array([embeddings_dict[path] for path in path_list])

#print(embeddings)
#a=np.array(issame_list)

tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embs, issame_list)
print(accuracy)
np.mean(accuracy)