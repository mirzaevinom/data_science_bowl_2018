"""
Prediction part of my solution to The 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.

author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""
from config import *
from model import log
from train import train_validation_split, KaggleDataset

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from skimage.measure import find_contours
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, erosion


import random
import pandas as pd
from metrics import mean_iou
from tqdm import tqdm

plt.switch_backend('agg')


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(masks, height, width):

    if masks.sum() < 1:
        masks = np.zeros([height, width, 1])
        # print('no masks')
        masks[0, 0, 0] = 1

    if np.any(masks.sum(axis=-1) > 1):
        print('Overlap', masks.shape)

    for mm in range(masks.shape[-1]):
        yield rle_encoding(masks[..., mm].astype(np.int32))


def plot_boundary(image, true_masks=None, pred_masks=None, ax=None):
    """
    Plots provided boundaries of nuclei for a given image.
    """
    if ax is None:
        n_rows = 1
        n_cols = 1

        fig = plt.figure(figsize=[4*n_cols, int(4*n_rows)])
        gs = gridspec.GridSpec(n_rows, n_cols)

        ax = fig.add_subplot(gs[0])

    ax.imshow(image)
    if true_masks is not None:
        for i in range(true_masks.shape[-1]):
            contours = find_contours(true_masks[..., i], 0.5, fully_connected='high')

            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
    if pred_masks is not None:
        for i in range(pred_masks.shape[-1]):
            contours = find_contours(pred_masks[..., i], 0.5, fully_connected='high')

            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)  # aspect ratio of 1


def plot_train(train_path='../data/stage1_train/'):
    """
    Plot and save true boundaries of nuclei for each training image
    """

    if not os.path.isdir('../train_images'):
        os.mkdir('../train_images')

    df = pd.read_csv('../data/train_df.csv')
    df = df.set_index('img_id')

    train_ids = os.listdir(train_path)
    dataset_train = KaggleDataset()
    dataset_train.load_shapes(train_ids, train_path)
    dataset_train.prepare()

    image_ids = dataset_train.image_ids
    for mm, image_id in tqdm(enumerate(image_ids)):

        image = dataset_train.load_image(image_id, color='RGB')
        masks, class_ids = dataset_train.load_mask(image_id)
        fig = plot_boundary(image, true_masks=masks)
        fig.savefig('../train_images/train_'+str(mm) +
                    '.png', bbox_inches='tight')
        plt.close()


def get_model(config, model_path=None):

    """
    Loads and returns MaskRCNN model for a given config and weights.
    """
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if model_path is None:
        model_path = model.find_last()[1]
        try:
            os.rename(model_path, model_path)
            print('Access on file ' + model_path + ' is available!')
            from shutil import copyfile
            dst = '../data/mask_rcnn_temp.h5'
            copyfile(model_path, dst)
            model_path = dst
        except OSError as e:
            print('Access-error on file "' + model_path + '"! \n' + str(e))

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)

    model.load_weights(model_path, by_name=True)

    return model


def ensemble_prediction(model, config, image):

    """ Test time augmentation method using non-maximum supression"""

    masks = []
    scores = []
    boxes = []

    results = {}

    result = model.detect([image], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    masks.append(result['masks'])
    scores.append(result['scores'])
    boxes.append(extract_bboxes(result['masks']))

    temp_img = np.fliplr(image)
    result = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    mask = np.fliplr(result['masks'])
    masks.append(mask)
    scores.append(result['scores'])
    boxes.append(extract_bboxes(mask))

    temp_img = np.flipud(image)
    result = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    mask = np.flipud(result['masks'])
    masks.append(mask)
    scores.append(result['scores'])
    boxes.append(extract_bboxes(mask))

    angle = np.random.choice([1, -1])
    temp_img = np.rot90(image, k=angle, axes=(0, 1))
    result = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    mask = np.rot90(result['masks'], k=-angle, axes=(0, 1))
    masks.append(mask)
    scores.append(result['scores'])
    boxes.append(extract_bboxes(mask))

    masks = np.concatenate(masks, axis=-1)
    scores = np.concatenate(scores, axis=-1)
    boxes = np.concatenate(boxes, axis=0)

    # config.DETECTION_NMS_THRESHOLD)
    keep_ind = non_max_suppression(boxes, scores, 0.1)
    masks = masks[:, :, keep_ind]
    scores = scores[keep_ind]

    results['masks'] = masks
    results['scores'] = scores

    return results


def cluster_prediction(model, config, image):

    """ Test time augmentation method using bounding box IoU"""
    # from utils import non_max_suppression, extract_bboxes, compute_overlaps
    height, width = image.shape[:2]

    # Predict masks on actual image
    result1 = model.detect([image], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    # Handles no mask predictions
    if result1['masks'].shape[0] == 0:
        result1['masks'] = np.zeros([height, width, 1])
        result1['masks'][0, 0, 0] = 1
        result1['scores'] = np.ones(1)

    # Predict masks on LR flipped image
    temp_img = np.fliplr(image)
    result2 = model.detect([temp_img], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]
    result2['masks'] = np.fliplr(result2['masks'])

    # Handles no mask predictions
    if result2['masks'].shape[0] == 0:
        result2['masks'] = np.zeros([height, width, 1])
        result2['masks'][0, 0, 0] = 1
        result2['scores'] = np.ones(1)

    # Compute IoU on masks
    overlaps = utils.compute_overlaps_masks(result1['masks'], result2['masks'])

    for mm in range(overlaps.shape[0]):

        if np.max(overlaps[mm]) > 0.1:
            ind = np.argmax(overlaps[mm])
            mask = result1['masks'][:, :, mm] + result2['masks'][:, :, ind]
            result1['masks'][:, :, mm] = (mask > 0).astype(np.uint8)
            # result1['scores'][mm] = 0.5*(result1['scores'][mm]+result2['scores'][ind])
        else:
            result1['masks'][:, :, mm] = 0

    return result1


def postprocess_masks(result, image, min_nuc_size=10):

    """Clean overlaps between bounding boxes, fill small holes, smooth boundaries"""

    height, width = image.shape[:2]

    # If there is no mask prediction do the following
    if result['masks'].shape[0] == 0:
        result['masks'] = np.zeros([height, width, 1])
        result['masks'][0, 0, 0] = 1
        result['scores'] = np.ones(1)
        result['class_ids'] = np.ones(1)

    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_nuc_size)[0]
    if len(keep_ind) < result['masks'].shape[-1]:
        # print('Deleting',len(result['masks'])-len(keep_ind), ' empty result['masks']')
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]
        result['rois'] = result['rois'][keep_ind]
        result['class_ids'] = result['class_ids'][keep_ind]

    sort_ind = np.argsort(result['scores'])[::-1]
    result['masks'] = result['masks'][..., sort_ind]
    overlap = np.zeros([height, width])

    # Removes overlaps from masks with lower score
    for mm in range(result['masks'].shape[-1]):
        # Fill holes inside the mask
        mask = binary_fill_holes(result['masks'][..., mm]).astype(np.uint8)
        # Smoothen edges using dilation and erosion
        mask = erosion(dilation(mask))
        # Delete overlaps
        overlap += mask
        mask[overlap > 1] = 0
        out_label = label(mask)
        # Remove all the pieces if there are more than one pieces
        if out_label.max() > 1:
            mask[()] = 0

        result['masks'][..., mm] = mask

    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_nuc_size)[0]
    if len(keep_ind) < result['masks'].shape[-1]:
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]
        result['rois'] = result['rois'][keep_ind]
        result['class_ids'] = result['class_ids'][keep_ind]

    return result


def eval_n_plot_val(model, config, dataset_val, save_plots=False):

    scores = []
    image_ids = dataset_val.image_ids

    for mm, image_id in tqdm(enumerate(image_ids)):
        # Load image and ground truth data
        image = dataset_val.load_image(image_id,  color=config.IMAGE_COLOR)
        gt_mask, gt_class_id = dataset_val.load_mask(image_id)

        img_name = dataset_val.image_info[image_id]['img_name']

        # result = ensemble_prediction(model, config, image)
        result = cluster_prediction(model, config, image)
        # result = model.detect([image], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]

        # Clean overlaps and apply some post-processing
        result = postprocess_masks(result, image)
        # If there is no masks then try to predict on scaled image
        if result['masks'].sum() < 2:
            H, W = image.shape[:2]
            scaled_img = np.zeros([4*H, 4*W, 3], np.uint8)
            scaled_img[:H, :W] = image
            result = cluster_prediction(model, config, scaled_img)
            result['masks'] = result['masks'][:H, :W]
            result = postprocess_masks(result, image)

        pred_box, pred_class_id, pred_score, pred_mask = result['rois'], result['class_ids'], \
            result['scores'], result['masks']

        gt_box = utils.extract_bboxes(gt_mask)
        # Compute IoU scores for ground truth and predictions
        iou = utils.compute_ap_range(gt_box, gt_class_id, gt_mask,
                                     pred_box, pred_class_id, pred_score, pred_mask,
                                     iou_thresholds=None, verbose=0)
        # iou = mean_iou(gt_mask, pred_masks)
        if save_plots:

            fig = plt.figure()
            gs = gridspec.GridSpec(1, 1)
            plot_boundary(image, true_masks=gt_mask, pred_masks=pred_mask,
                          ax=fig.add_subplot(gs[0]))

            fig.savefig('../images/validation_'+str(mm)+'.png', bbox_inches='tight')
            plt.close()

        scores.append(iou)
        if (mm+1) % 10 == 0:
            print('Mean IoU for', mm+1, 'imgs', np.mean(scores))

    print("Mean IoU: ", np.mean(scores))


def pred_n_plot_test(model, config, test_path='../data/stage2_test_final/', save_plots=False):
    """
    Predicts nuclei for each image, draws the boundaries and saves in images folder.

    """
    # Create images folder if doesn't exist.
    if not os.path.isdir('../images'):
        os.mkdir('../images')

    # Load test dataset
    test_ids = os.listdir(test_path)
    dataset_test = KaggleDataset()
    dataset_test.load_shapes(test_ids, test_path)
    dataset_test.prepare()

    new_test_ids = []
    rles = []

    # No masks prediction counter
    no_masks = 0

    for mm, image_id in tqdm(enumerate(dataset_test.image_ids)):
        # Load the image
        image = dataset_test.load_image(image_id, color=config.IMAGE_COLOR)

        # Image name for submission rows.
        image_id = dataset_test.image_info[image_id]['img_name']
        height, width = image.shape[:2]

        # result = ensemble_prediction(model, config, image)
        result = cluster_prediction(model, config, image)
        # result = model.detect([image], verbose=0, mask_threshold=config.DETECTION_MASK_THRESHOLD)[0]

        # Clean overlaps and apply some post-processing
        result = postprocess_masks(result, image)
        # If there is no masks then try to predict on scaled image
        if result['masks'].sum() < 2:
            H, W = image.shape[:2]
            scaled_img = np.zeros([4*H, 4*W, 3], np.uint8)
            scaled_img[:H, :W] = image
            result = cluster_prediction(model, config, scaled_img)
            result['masks'] = result['masks'][:H, :W]
            result = postprocess_masks(result, image)

        if result['masks'].sum() < 1:
            no_masks += 1

        # RLE encoding
        rle = list(prob_to_rles(result['masks'], height, width))
        new_test_ids.extend([image_id] * len(rle))
        rles.extend(rle)
        # if (mm+1) % 100 == 0:
        #     print('Number of completed images', mm+1)

        if save_plots:
            fig = plt.figure()
            gs = gridspec.GridSpec(1, 1)
            plot_boundary(image, true_masks=None, pred_masks=result['masks'],
                          ax=fig.add_subplot(gs[0]))

            fig.savefig('../images/' + str(image_id)+'.png', bbox_inches='tight')
            plt.close()

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    fname = '../data/' + test_path.split('/')[-2]+'_submission.csv'
    sub.to_csv(fname, index=False)

    print('Number of rows:', len(sub))
    print('No mask prediction for', no_masks, 'images')


if __name__ == '__main__':

    import time

    start = time.time()

    # Create model configuration in inference mode
    config = KaggleBowlConfig()
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = 1
    config.display()

    # Predict using the last weights in training directory
    model = get_model(config)

    # Predict using pre-trained weights
    # model = get_model(config, model_path='../data/kaggle_bowl.h5')


    # Ininitialize validation dataset
    train_path = '../data/stage1_train/'
    train_list, val_list = train_validation_split(
        train_path, seed=11, test_size=0.1)
    dataset_val = KaggleDataset()
    dataset_val.load_shapes(val_list, train_path)
    dataset_val.prepare()

    # initialize stage 1  testing dataset
    dataset_val = KaggleDataset()
    val_path = '../data/stage1_test/'
    val_list = os.listdir(val_path)
    dataset_val.load_shapes(val_list, val_path)
    dataset_val.prepare()
    # Evaluate the model performance and plot boundaries for the predictions
    eval_n_plot_val(model, config, dataset_val, save_plots=True)

    # Predict and plot boundaries for stage1 test
    pred_n_plot_test(model, config, test_path='../data/stage1_test/', save_plots=True)
    # Predict and plot boundaries for stage2 test
    pred_n_plot_test(model, config, test_path='../data/stage2_test_final/', save_plots=True)

    # Save supercomputer log file locally
    if 'PBS_JOBID' in os.environ.keys():
        job_id = os.environ['PBS_JOBID'][:7]
        fileList = list(filter(lambda x: job_id in x, os.listdir('./')))
        os.rename(fileList[0], 'log.txt')

    print('Elapsed time', round((time.time() - start)/60, 1), 'minutes')
