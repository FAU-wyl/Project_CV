# -*- coding: utf-8 -*-
import os
import shlex
import argparse
from tqdm import tqdm

# Python3 里读取/保存 pickle 缓存文件。老师给的 skeleton 里用的是 cPickle 风格。
# Reading/saving pickle cache files in Python3. The skeleton provided by the teacher used cPickle style.
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
import numpy as np
import cv2

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-str', '--suffix_train',
                        default='.png',
                        help='only chose those images with a specific suffix')
    parser.add_argument('-ste', '--suffix_test',
                        default='.jpg',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--to_binary', action='store_true',
                       help='use OTSU binarization')

    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    # 是否使用 VLAD 常见的 signed square-root power normalization。
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    # 一次性计算 base VLAD 和 powernorm VLAD，避免重复提取 3600 张测试图的 SIFT。
    parser.add_argument('--both', action='store_true',
                        help='compute and evaluate base VLAD and powernorm VLAD in one pass')
    # 视觉词数量 K。作业要求通常是 K=100，因此默认设置为 100。
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='number of visual words for the codebook')
    # 从训练集局部 SIFT 描述子中最多抽多少个用于 KMeans。作业建议约 500k。
    parser.add_argument('--max_descriptors', default=500000, type=int,
                        help='maximum number of SIFT descriptors sampled for KMeans')
    # 为了控制 KMeans 前的 SIFT 提取耗时，默认只从 100 张训练图中抽描述子。
    parser.add_argument('--max_dictionary_files', default=100, type=int,
                        help='number of training files sampled for KMeans; use 0 for all files')
    # E-SVM 很慢，所以默认不跑；需要时显式加 --esvm。
    parser.add_argument('--esvm', action='store_true',
                        help='also run exemplar SVM after VLAD evaluation')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # 读取标签文件。每一行格式是：图像名 作者/类别ID。
    # Read label file. Each line format is: image_name author/class_id.
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # 根据标签文件生成真实图像路径，同时保留每张图的 writer label。
    all_files = []
    labels = []
    for line in all_lines:
        # shlex 可以兼容带引号/转义空格的文件名。
        # shlex can handle filenames with quotes/escaped spaces.
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # 去掉标签文件里可能已有的后缀，再统一拼接当前数据集需要的后缀。
        # Remove any suffixes that may already exist in the label file, then uniformly concatenate the suffix needed for the current dataset.
        # 这里不用 os.path.splitext，因为部分文件名中间也可能含有 '.'。
        # We don't use os.path.splitext here because some filenames may contain '.' in the middle.
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # 拼出完整路径，例如 training folder + image id + .png。
        # Concatenate the full path, e.g., training_folder + image_id + .png.
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors, max_files=100):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # 训练 codebook 不需要用完所有 SIFT；随机抽一部分图像和描述子就足够。
    # Training the codebook doesn't require using all SIFT; randomly sampling some images and descriptors is sufficient.
    files = np.array(files)
    if max_files is not None and max_files > 0:
        max_files = min(max_files, len(files))
        indices = np.random.permutation(len(files))[:max_files]
        files = files[indices]
    else:
        # max_files<=0 时使用全部训练图，但仍然打乱顺序，避免标签文件排序带来的偏差。
        # When max_files<=0, use all training images, but still shuffle the order to avoid bias from label file sorting.
        files = files[np.random.permutation(len(files))]
   
    # 粗略限制每张图贡献的 SIFT 数量，避免少数大图主导 KMeans。
    # Roughly limit the number of SIFT descriptors contributed by each image to prevent a few large images from dominating KMeans.
    max_descs_per_file = max(1, int(np.ceil(max_descriptors / len(files))))

    descriptors = []
    for i in tqdm(range(len(files)), desc='sample SIFT'):
        # 使用 RootSIFT：先提取 SIFT，再做 Hellinger normalization。
        # Use RootSIFT: extract SIFT first, then apply Hellinger normalization.
        desc = computeDescs(files[i], True, True)
        if len(desc) == 0:
            continue
            
        # 从当前图片的全部 SIFT 中随机抽样。
        # Randomly sample from all SIFT descriptors of the current image.
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    if len(descriptors) == 0:
        raise RuntimeError('no descriptors extracted from the selected files')

    # 合并成 Qx128 的矩阵，作为 MiniBatchKMeans 的输入。
    # Concatenate into a Qx128 matrix as input to MiniBatchKMeans.
    descriptors = np.concatenate(descriptors, axis=0).astype(np.float32)
    if len(descriptors) > max_descriptors:
        # 如果由于 ceil 导致总数超过上限，再做一次全局随机截断。
        # If the total exceeds the limit due to ceil, perform one more global random truncation.
        indices = np.random.choice(len(descriptors), max_descriptors, replace=False)
        descriptors = descriptors[indices]
    return descriptors

def toBinary(mask):
    if mask is None:
        raise ValueError('cannot binarize an empty image')

    # 如果图片已经是 0/255 二值图，就直接返回；否则做 Otsu 二值化。
    # If the image is already a 0/255 binary image, return directly; otherwise, perform Otsu binarization.
    if mask[mask==255].sum() != np.sum(mask):
        # 有些二值图可能是 0/1，需要转成 OpenCV 常用的 0/255。
        # Some binary images may be 0/1 and need to be converted to OpenCV's standard 0/255.
        if mask[mask==1].sum() == mask.sum():
            mask *= 255
        else: # 普通灰度图：用 Otsu 自动找阈值。For regular grayscale images: use Otsu to automatically find the threshold.
           ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return mask

def computeDescs(fname, norm_hellinger=False, to_binary=False):
    # 灰度读取手写图；SIFT 本身只需要单通道图像。
    # Read the handwritten image in grayscale; SIFT only requires single-channel images.
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(fname)

    if to_binary:
        # 数据本身是 binary 版本；这里保留开关，确保输入统一为二值形式。
        # The data is itself a binary version; here we keep the switch to ensure input is uniformly in binary form.
        img = toBinary(img)

    # OpenCV SIFT：先检测关键点，再在这些关键点上计算 128 维 descriptor。
    # OpenCV SIFT: first detect keypoints, then compute 128-dimensional descriptors at these keypoints.
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)
    for keypoint in keypoints:
        # 作业要求将 SIFT keypoint 的方向固定为 0，不使用旋转不变性。
        # The assignment requires fixing the direction of SIFT keypoints to 0, not using rotation invariance.
        keypoint.angle = 0.0

    keypoints, descriptors = sift.compute(img, keypoints)
    if descriptors is None:
        # 极端情况下图片没有关键点，返回空的 0x128 矩阵，后续 VLAD 会变成零向量。
        # In extreme cases when an image has no keypoints, return an empty 0x128 matrix, which will become a zero vector in subsequent VLAD.
        return np.empty((0, 128), dtype=np.float32)

    descriptors = descriptors.astype(np.float32)
    if norm_hellinger:
        # Hellinger/RootSIFT normalization：
        # Hellinger/RootSIFT normalization:
        # 先做 L1 normalize，再对每个维度开方，通常比原始 SIFT 更适合欧氏距离/KMeans。
        # First perform L1 normalization, then take the square root of each dimension, which is usually more suitable for Euclidean distance/KMeans than original SIFT.
        eps = np.finfo(np.float32).eps
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
    
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # 用训练集抽样出来的 SIFT 描述子训练视觉词典 codebook。
    # Train the visual word codebook using SIFT descriptors sampled from the training set.
    # MiniBatchKMeans 比普通 KMeans 快很多，适合 500k 这种规模。
    # MiniBatchKMeans is much faster than standard KMeans and is suitable for this scale of ~500k.
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=max(1000, n_clusters * 20),
        n_init=3,
        random_state=42,
        verbose=0,
    )
    kmeans.fit(descriptors)
    # cluster_centers_ 的形状是 Kx128，即 K 个视觉词中心。
    # The shape of cluster_centers_ is Kx128, i.e., K visual word centers.
    return kmeans.cluster_centers_.astype(np.float32)

def nearest_cluster_indices(descriptors, clusters):
    """返回每个局部 descriptor 最近的视觉词编号。"""
    descriptors = descriptors.astype(np.float32, copy=False)
    clusters = clusters.astype(np.float32, copy=False)

    # 用 ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c 计算距离。
    # Calculate distance using ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c.
    # 这样不用显式创建 T x K x D 的三维 residual 张量，内存更稳。
    # This avoids explicitly creating a 3D residual tensor of T x K x D, making memory more stable.
    desc_norms = np.sum(descriptors * descriptors, axis=1, keepdims=True)
    cluster_norms = np.sum(clusters * clusters, axis=1)
    dists = desc_norms + cluster_norms[None, :] - 2.0 * descriptors.dot(clusters.T)
    return np.argmin(dists, axis=1)

def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    if len(descriptors) == 0:
        return np.empty((0, len(clusters)), dtype=np.float32)

    # 对每个 SIFT descriptor 找最近的 cluster center。
    # Find the nearest cluster center for each SIFT descriptor.
    nearest = nearest_cluster_indices(descriptors, clusters)

    # hard assignment：每个 descriptor 只属于最近的一个视觉词，用 one-hot 表示。
    # Hard assignment: each descriptor belongs only to the nearest visual word, represented as one-hot.
    assignment = np.zeros( (len(descriptors), len(clusters)), dtype=np.float32 )
    assignment[np.arange(len(descriptors)), nearest] = 1.0

    return assignment

def raw_vlad(desc, mus):
    K, D = mus.shape
    if len(desc) == 0:
        # 没有 SIFT 时无法累计 residual，返回全零 VLAD 向量。
        # Without SIFT, cannot accumulate residuals; return an all-zero VLAD vector.
        return np.zeros((D * K), dtype=np.float32)

    # VLAD 核心：每个 descriptor 分配到最近视觉词，然后累计 descriptor-center 的残差。
    # VLAD core: each descriptor is assigned to the nearest visual word, then accumulate the residuals of descriptor-center.
    nearest = nearest_cluster_indices(desc, mus)
    residuals = desc - mus[nearest]
    f_enc = np.zeros((K, D), dtype=np.float32)
    # np.add.at 会把同一个 cluster 的 residual 全部加到对应行。
    # np.add.at adds all residuals of the same cluster to the corresponding row.
    np.add.at(f_enc, nearest, residuals)
    # KxD 拉平成 K*D；K=100, D=128 时就是 12800 维全局图像特征。
    # Flatten KxD into K*D; when K=100 and D=128, it becomes a 12800-dimensional global image feature.
    return f_enc.ravel()

def normalize_vlad(f_enc, powernorm):
    f_enc = f_enc.astype(np.float32, copy=True)
    if powernorm:
        # signed square-root：压低特别大的 residual 维度，减少 burstiness 的影响。
        # Signed square-root: suppress particularly large residual dimensions to reduce the effect of burstiness.
        f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

    # 最后做 L2 normalize，这样后续可以直接用 dot product 计算 cosine similarity。
    # Finally perform L2 normalization, so that subsequent cosine similarity can be calculated directly using dot product.
    norm = np.linalg.norm(f_enc)
    if norm > 0:
        f_enc /= norm
    return f_enc.astype(np.float32)

def vlad(files, mus, powernorm):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    encodings = []

    for f in tqdm(files, desc='VLAD'):
        # 每张图片先变成可变数量的 RootSIFT，再聚合成固定长度 VLAD。
        # Each image is first converted to a variable-length RootSIFT, then aggregated into a fixed-length VLAD.
        desc = computeDescs(f, True, True)
        f_enc = raw_vlad(desc, mus)
        encodings.append(normalize_vlad(f_enc, powernorm))

    # 返回 Nx(K*D) 矩阵，N 是图片数量。
    # Return an Nx(K*D) matrix, where N is the number of images.
    return np.vstack(encodings).astype(np.float32)

def vlad_both(files, mus):
    # 同一张图的 raw VLAD 只算一次，然后分别生成 base 和 powernorm 版本。
    # The raw VLAD of the same image is computed only once, then both base and powernorm versions are generated separately.
    # 这样跑 --both 时不用重复提取 SIFT，能节省很多时间。
    # This way when running --both, SIFT extraction doesn't need to be repeated, saving a lot of time.
    enc_base = []
    enc_power = []

    for f in tqdm(files, desc='VLAD'):
        desc = computeDescs(f, True, True)
        f_enc = raw_vlad(desc, mus)
        enc_base.append(normalize_vlad(f_enc, False))
        enc_power.append(normalize_vlad(f_enc, True))

    return (
        np.vstack(enc_base).astype(np.float32),
        np.vstack(enc_power).astype(np.float32),
    )

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """
    # 对每个测试样本训练一个二分类器：
    # Train a binary classifier for each test sample:
    # 正样本是当前测试图，负样本是所有训练图。
    # Positive sample is the current test image, negative samples are all training images.
    y = np.hstack(([1], -np.ones(len(encs_train), dtype=np.int32)))

    def loop(i):
        # 训练当前测试图的 exemplar SVM。
        # Train an exemplar SVM for the current test image.
        x = np.vstack((encs_test[i:i+1], encs_train))
        clf = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        clf.fit(x, y)
        # 取 SVM 的权重向量 coef_ 作为这张测试图的新 embedding。
        # Use the SVM's weight vector coef_ as the new embedding for this test image.
        transformed = clf.coef_.astype(np.float32)
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed /= norm
        return transformed

    # E-SVM 需要给每张测试图都训练一次 SVM，非常慢；优先用 parmap 并行。
    # E-SVM requires training an SVM for each test image, which is very slow; preferably use parmap for parallelization.
    # 如果没有 parmap 依赖，就退回普通 map，保证脚本仍然能跑。
    # If parmap dependency is not available, fall back to standard map to ensure the script still works.
    # try:
    #     from parmap import parmap
    #     new_encs = list(parmap(loop, tqdm(range(len(encs_test)), desc='E-SVM')))
    # except ImportError:
    #     new_encs = list(map(loop, tqdm(range(len(encs_test)), desc='E-SVM')))
    new_encs = list(map(loop, tqdm(range(len(encs_test)), desc='E-SVM')))
    new_encs = np.concatenate(new_encs, axis=0)
    # 返回所有测试图的 E-SVM embedding。
    # Return the E-SVM embeddings for all test images.
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # 先保险地再做一次 L2 normalize，然后用 1-dot 作为 cosine distance。
    # First, to be safe, perform L2 normalization again, then use 1-dot as cosine distance.
    encs = encs.astype(np.float32, copy=False)
    norms = np.linalg.norm(encs, axis=1, keepdims=True)
    encs = encs / np.maximum(norms, np.finfo(np.float32).eps)
    dists = 1.0 - encs.dot(encs.T)
    # 把自己到自己的距离设成最大值，避免检索第一名永远是 query 自己。
    # Set the distance from each image to itself to the maximum value to avoid the first result always being the query itself.
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # 每一行对应一个 query，把数据库图片按距离从小到大排序。
    # Each row corresponds to a query; sort database images by distance in ascending order.
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        # query r 的 relevant 图片，就是 label 和 query 相同的图片。
        # Relevant images for query r are images whose label is the same as the query.
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                # 每遇到一个 relevant，就记录当前位置的 precision。
                # Each time a relevant image is found, record the precision at the current position.
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        # 当前 query 的 AP；所有 query 的 AP 平均后就是 mAP。
        # The AP for the current query; the average AP of all queries is the mAP.
        avg_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    top1 = float(correct) / n_encs
    print('Top-1 accuracy: {} - mAP: {}'.format(top1, mAP))
    return top1, mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    # 固定随机种子，保证 descriptor 抽样和 KMeans 初始化尽量可复现。
    # Fix the random seed to ensure descriptor sampling and KMeans initialization are reproducible.
    np.random.seed(42)

    # 这些路径参数是主流程必须的，缺任何一个都无法定位数据或标签。
    # These path parameters are required for the main process; missing any one makes it impossible to locate data or labels.
    for required_arg in ['in_train', 'in_test', 'labels_train', 'labels_test']:
        if getattr(args, required_arg) is None:
            raise ValueError('--{} is required'.format(required_arg))
   
    # a) 读取训练集文件列表，用训练集 SIFT 建 codebook。
    # a) Read the training set file list and build a codebook using training set SIFT descriptors.
    files_train, labels_train = getFiles(args.in_train, args.suffix_train,
                                         args.labels_train)
    assert (len(files_train) == len(labels_train))
    print('#train: {}'.format(len(files_train)))

    missing_train = [f for f in files_train if not os.path.exists(f)]
    if missing_train:
        raise FileNotFoundError('missing training file, e.g. {}'.format(missing_train[0]))

    # codebook 缓存名包含关键参数，避免 K 或抽样数量变了却误用旧词典。
    # The codebook cache name includes key parameters to avoid accidentally using an old codebook when K or sampling size changes.
    dict_fname = 'mus_k{}_d{}_f{}.pkl.gz'.format(
        args.n_clusters, args.max_descriptors, args.max_dictionary_files)
   
    if not os.path.exists(dict_fname) or args.overwrite:
        # 第一次运行时：从训练图随机抽 RootSIFT，再训练 KMeans codebook。
        # On first run: randomly sample RootSIFT from training images, then train a KMeans codebook.
        descriptors = loadRandomDescriptors(
            files_train,
            args.max_descriptors,
            args.max_dictionary_files,
        )
        print('> computed/loaded {} descriptors'.format(len(descriptors)))

        # KMeans 的 cluster centers 就是 VLAD 的视觉词典。
        # KMeans cluster centers are the visual word codebook for VLAD.
        print('> compute dictionary')
        mus = dictionary(descriptors, args.n_clusters)
        with gzip.open(dict_fname, 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        # 后续重复实验直接加载 codebook，节省 KMeans 时间。
        # For subsequent experiments, load the codebook directly to save KMeans time.
        print('> load dictionary from {}'.format(dict_fname))
        with gzip.open(dict_fname, 'rb') as f:
            mus = cPickle.load(f)

  
    # b/c/d) 对测试集图片做 VLAD encoding，并评估 retrieval。
    # b/c/d) Perform VLAD encoding on test set images and evaluate retrieval.
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix_test,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))

    missing_test = [f for f in files_test if not os.path.exists(f)]
    if missing_test:
        raise FileNotFoundError('missing test file, e.g. {}'.format(missing_test[0]))

    # 分别缓存 base VLAD 和 powernorm VLAD，文件名里记录 K。
    # Cache base VLAD and powernorm VLAD separately, with K recorded in the filename.
    base_fname = 'enc_test_k{}_base.pkl.gz'.format(args.n_clusters)
    power_fname = 'enc_test_k{}_powernorm.pkl.gz'.format(args.n_clusters)

    if args.both:
        # --both：一次读取图片，同时生成两种 normalization 的 VLAD。
        # --both: read images once and generate both base and powernorm normalized VLAD simultaneously.
        if (not os.path.exists(base_fname) or not os.path.exists(power_fname)
                or args.overwrite):
            enc_base, enc_power = vlad_both(files_test, mus)
            with gzip.open(base_fname, 'wb') as fOut:
                cPickle.dump(enc_base, fOut, -1)
            with gzip.open(power_fname, 'wb') as fOut:
                cPickle.dump(enc_power, fOut, -1)
        else:
            print('> load test encodings from {} and {}'.format(base_fname, power_fname))
            with gzip.open(base_fname, 'rb') as f:
                enc_base = cPickle.load(f)
            with gzip.open(power_fname, 'rb') as f:
                enc_power = cPickle.load(f)

        # 分别评估基础 VLAD 和 powernorm VLAD，方便报告中直接比较。
        # Evaluate base VLAD and powernorm VLAD separately for convenient comparison in reports.
        print('> evaluate base VLAD')
        evaluate(enc_base, labels_test)
        print('> evaluate VLAD + powernorm')
        evaluate(enc_power, labels_test)
        raise SystemExit(0)

    suffix = 'powernorm' if args.powernorm else 'base'
    fname = power_fname if args.powernorm else base_fname
    if not os.path.exists(fname) or args.overwrite:
        # 单独模式：只计算用户指定的 base 或 powernorm 版本。
        # Standalone mode: compute only the user-specified base or powernorm version.
        enc_test = vlad(files_test, mus, args.powernorm)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        # 如果已有缓存，直接加载再评估。
        # If cache exists, load directly and evaluate.
        print('> load test encodings from {}'.format(fname))
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)

    # 在测试集内部做检索评估：每张测试图作为 query，其他测试图作为数据库。
    # Perform retrieval evaluation within the test set: each test image as query, other test images as database.
    print('> evaluate {}'.format(suffix))
    evaluate(enc_test, labels_test)

    if args.esvm:
        # 可选 e) Exemplar SVM：需要训练集 VLAD 作为负样本。
        # Optional e) Exemplar SVM: requires training set VLAD as negative samples.
        print('> compute VLAD for train (for E-SVM)')
        fname = 'enc_train_k{}_{}.pkl.gz'.format(args.n_clusters, suffix)
        if not os.path.exists(fname) or args.overwrite:
            enc_train = vlad(files_train, mus, args.powernorm)
            with gzip.open(fname, 'wb') as fOut:
                cPickle.dump(enc_train, fOut, -1)
        else:
            print('> load train encodings from {}'.format(fname))
            with gzip.open(fname, 'rb') as f:
                enc_train = cPickle.load(f)

        # 用 E-SVM 权重替换原测试 embedding，再重新做 retrieval 评估。
        # Replace the original test embedding with E-SVM weights and re-evaluate retrieval.
        print('> esvm computation')
        enc_test = esvm(enc_test, enc_train, args.C)

        print('> evaluate E-SVM')
        evaluate(enc_test, labels_test)
