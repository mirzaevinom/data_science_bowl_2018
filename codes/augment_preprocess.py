"""
Augmentation and preprocessing functions of my solution to
The 2018 Data Science Bowl: https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.

author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""

from config import *


from xml.dom import minidom
import cv2
import h5py
from skimage.measure import label

def fix_crop_transform(image, mask, x, y, w, h):
    """
    Helper function for random_crop_transform.

    Modified from the codes provided in the following thread:
    https://www.kaggle.com/c/data-science-bowl-2018/discussion/49692
    """
    H, W = image.shape[:2]
    assert(H >= h)
    assert(W >= w)

    if (x == -1 & y == -1):
        x = (W-w)//2
        y = (H-h)//2

    if (x, y, w, h) != (0, 0, W, H):
        image = image[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

    return image, mask


def random_crop_transform(image, mask, w, h):
    """
    Randomly crops given image and mask to the given w and h.
    Modified from the codes provided in the following thread:
    https://www.kaggle.com/c/data-science-bowl-2018/discussion/49692
    """

    H, W = image.shape[:2]

    if H != h:
        y = np.random.choice(H-h)
    else:
        y = 0

    if W != w:
        x = np.random.choice(W-w)
    else:
        x = 0

    return fix_crop_transform(image, mask, x, y, w, h)


def relabel_multi_mask(multi_mask):
    """
    Relabels masks after random rotation, scaling and shifting.

    Modified from the codes provided in the following thread:
    https://www.kaggle.com/c/data-science-bowl-2018/discussion/49692
    """

    data = multi_mask
    data = data[:, :, np.newaxis]
    unique_color = set(tuple(v) for m in data for v in m)

    H, W = data.shape[:2]
    multi_mask = np.zeros((H, W), np.int32)
    for color in unique_color:

        if color == (0,):
            continue

        mask = (data == color).all(axis=2)
        out_label = label(mask)

        index = [out_label != 0]
        multi_mask[index] = out_label[index]+multi_mask.max()

    return multi_mask


def random_shift_scale_rotate_transform(image, mask,
                                        shift_limit=[-0.0625, 0.0625], scale_limit=[1/1.2, 1.2],
                                        rotate_limit=[-15, 15], borderMode=cv2.BORDER_REFLECT_101):

    """
    Random rotations, scaling and shifting for image augmentation.

    Modified from the codes provided in the following thread:
    https://www.kaggle.com/c/data-science-bowl-2018/discussion/49692
    """

    height, width, channel = image.shape

    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
    scale = np.random.uniform(scale_limit[0], scale_limit[1])
    sx = scale
    sy = scale
    dx = round(np.random.uniform(shift_limit[0], shift_limit[1])*width)
    dy = round(np.random.uniform(shift_limit[0], shift_limit[1])*height)

    cc = np.cos(angle/180*np.pi)*(sx)
    ss = np.sin(angle/180*np.pi)*(sy)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx, height/2+dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    mask = mask.astype(np.float32)
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                               borderMode=borderMode, borderValue=(0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = mask.astype(np.int32)
    mask = label(mask)
    # mask = relabel_multi_mask(mask)

    return image, mask


def clean_masks(mask, min_mask_area=0):
    """
    Removes masks with small area and adjust class_ids accordingly.
    """

    height, width = mask.shape[:2]

    keep_ind = np.where(np.sum(mask, axis=(0, 1)) > min_mask_area)[0]

    if len(keep_ind) > 0:
        mask = mask[:, :, keep_ind]
        class_ids = np.ones(mask.shape[-1], np.uint8)
    else:
        class_ids = np.zeros(mask.shape[-1], np.uint8)
        mask = np.ones([height, width, 1])

    return mask, class_ids



def data_to_array(train_path='../data/stage1_train/'):
    """ Save masks as h5 files to make trainig faster"""

    train_ids = list(
        filter(lambda x: ('mosaic' not in x) and ('TCGA' not in x), os.listdir(train_path)))

    for i, id_ in enumerate(train_ids):
        path = train_path + id_

        if os.path.exists(path + '/images/' + id_ + '.h5'):
            os.remove(path + '/images/' + id_ + '.h5')
        if os.path.exists(path + '/masks/' + id_ + '.h5'):
            os.remove(path + '/masks/' + id_ + '.h5')

        mask = []
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            if 'png' in mask_file:
                mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
                mask_ = np.where(mask_ > 128, 1, 0)

                mask_ = binary_fill_holes(mask_).astype(np.int32)
                if np.sum(mask_) >= 1:
                    mask.append(np.squeeze(mask_))

        mask = np.stack(mask, axis=-1)
        mask = mask.astype(np.uint8)

        fname = path + '/masks/' + id_ + '.h5'
        with h5py.File(fname, "w") as hf:
            hf.create_dataset("arr", data=mask)

        if (i+1) % 20 == 0:
            print(i+1)


def combine_images(img_list):
    """ Combines imgs using indexes as follows:
        0 1
        2 3
    """
    up = np.hstack(img_list[:2])
    down = np.hstack(img_list[2:])
    full = np.vstack([up, down])
    return full


def combine_masks(mask_list):
    """ Combines masks using indexes as follows:
        0 1
        2 3
    modified from
    https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/rebuild_mosaics.py
    """
    H = mask_list[0].shape[0]+mask_list[2].shape[0]
    W = mask_list[0].shape[1]+mask_list[1].shape[1]

    row, col, n_mask = mask_list[0].shape
    temp = np.zeros([H, W, n_mask], np.uint8)
    temp[:row, :col] = mask_list[0]
    mask_list[0] = temp.copy()

    row, col, n_mask = mask_list[1].shape
    temp = np.zeros([H, W, n_mask], np.uint8)
    temp[:row, col:] = mask_list[1]
    mask_list[1] = temp.copy()

    row, col, n_mask = mask_list[2].shape
    temp = np.zeros([H, W, n_mask], np.uint8)
    temp[row:, :col] = mask_list[2]
    mask_list[2] = temp.copy()

    row, col, n_mask = mask_list[3].shape
    temp = np.zeros([H, W, n_mask], np.uint8)
    temp[row:, col:] = mask_list[3]
    mask_list[3] = temp.copy()

    mask_list = np.concatenate(mask_list, axis=-1)

    return mask_list.astype(np.uint8)


def map_layers_left_to_right(mask, center, left_half_idx, right_half_idx):
    """
    Selects two adjacent 1-pixel stripes on the edges of left and right halves, then calculates
    how many pixels touch each other for each layer pair. Returns a map of scores.
    modified from
    https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/rebuild_mosaics.py

    """
    result = np.zeros((left_half_idx.shape[0], right_half_idx.shape[0]))

    for left_i, left_id in enumerate(left_half_idx):
        for right_i, right_id in enumerate(right_half_idx):
            result[left_i, right_i] = \
                np.logical_and(
                    mask[:, center - 1, left_id],
                    mask[:, center, right_id]).sum()

    return result


def map_layers_top_to_bottom(mask, center, top_half_idx, bottom_half_idx):
    """
    Same as `map_layers_left_to_right`, but top to bottom.
    modified from
    https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/rebuild_mosaics.py

    """
    result = np.zeros((top_half_idx.shape[0], bottom_half_idx.shape[0]))

    for top_i, top_id in enumerate(top_half_idx):
        for bottom_i, bottom_id in enumerate(bottom_half_idx):
            result[top_i, bottom_i] = \
                np.logical_and(
                    mask[center - 1, :, top_id],
                    mask[center, :, bottom_id]).sum()

    return result


def merge_layers(mask, intersection_map, survivor_half_idx, merged_half_idx):
    """
    Merges `merged_half_idx` channels of the mask into `survivor_half_idx`. If no intersection
    was found between layers, layers are kept.
    modified from
    https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/rebuild_mosaics.py

    """
    if intersection_map.shape[0] == 0 or intersection_map.shape[1] == 0:
        return mask, 0

    merged_idx = []

    for i in range(survivor_half_idx.shape[0]):
        if intersection_map[i].max() == 0:
            continue

        survivor_id = survivor_half_idx[i]
        merged_id = merged_half_idx[intersection_map[i].argmax()]

        mask[:, :, survivor_id] = np.logical_or(
            mask[:, :, survivor_id], mask[:, :, merged_id])
        merged_idx.append(merged_id)

    return np.delete(mask, merged_idx, axis=-1), len(merged_idx)


def merge_layers_on_edges(mask):
    """
    First merges layers left halves to right halves, then top to bottom.
    modified from
    https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/rebuild_mosaics.py

    """
    lr_center = mask.shape[1] // 2
    left_half_idx = np.argwhere(
        mask[:, lr_center - 1, :].sum(axis=0) > 0).flatten()
    right_half_idx = np.argwhere(
        mask[:, lr_center, :].sum(axis=0) > 0).flatten()
    lr_map = map_layers_left_to_right(
        mask, lr_center, left_half_idx, right_half_idx)
    mask, deleted_layers_count_ltr = merge_layers(
        mask, lr_map, left_half_idx, right_half_idx)

    tb_center = mask.shape[0] // 2
    top_half_idx = np.argwhere(
        mask[tb_center - 1, :, :].sum(axis=0) > 0).flatten()
    bottom_half_idx = np.argwhere(
        mask[tb_center, :, :].sum(axis=0) > 0).flatten()
    tb_map = map_layers_top_to_bottom(
        mask, tb_center, top_half_idx, bottom_half_idx)
    mask, deleted_layers_count_ttb = merge_layers(
        mask, tb_map, top_half_idx, bottom_half_idx)

    return mask, deleted_layers_count_ltr + deleted_layers_count_ttb


def make_n_save_mosaic(train_path='../data/stage1_train/'):
    """
    Mosaic function is mostly identical to that of https://www.kaggle.com/bonlime/train-test-image-mosaic
    """
    df = pd.read_csv('../data/share_train_df.csv')
    df = df[~df['mosaic_position'].isnull()]
    df = df.sort_values('mosaic_idx')

    pos_dict = {'up_left': 0, 'up_right': 1, 'down_left': 2, 'down_right': 3}
    df['mosaic_position'] = df['mosaic_position'].replace(pos_dict)
    # df['HSV_CLUSTER'].value_counts()

    for mosaic_id in tqdm(df['mosaic_idx'].unique()):
        mini_df = df[df['mosaic_idx'] == mosaic_id]
        mini_df = mini_df.sort_values('mosaic_position')
        img_list = []
        mask_list = []

        for img_id, pos in mini_df[['img_id', 'mosaic_position']].values:

            path = train_path+img_id

            fname = path+'/images/'+img_id+'.png'
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img_list.append(img)

            path = path + '/masks/' + img_id + '.h5'
            with h5py.File(path, "r") as hf:
                mask = hf["arr"][()]

            mask_list.append(mask)

        path = train_path+'mosaic_id_'+str(mosaic_id)

        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(path+'/images/'):
            os.mkdir(path+'/images/')

        if not os.path.isdir(path+'/masks/'):
            os.mkdir(path+'/masks/')

        img_list = combine_images(img_list)
        fname = path+'/images/'+'mosaic_id_'+str(mosaic_id)+'.png'
        cv2.imwrite(fname, img_list)

        mask_list = combine_masks(mask_list)
        mask_list, _ = merge_layers_on_edges(mask_list)
        fname = path+'/masks/'+'mosaic_id_'+str(mosaic_id)+'.h5'
        with h5py.File(fname, "w") as hf:
            hf.create_dataset("arr", data=mask_list)


def split_image(img):
    """
    Split image into four small pieces.
    """
    imgheight, imgwidth = img.shape[:2]
    height = imgheight//2
    width = imgwidth//2

    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            yield img[i*height:(i+1)*height, j*width:(j+1)*width]


def preprocess_external_data(annot_path, tissue_path, train_path):
    """
    Preprocesses external data and puts them in training directory.
    Modified the codes provided in
    https://www.kaggle.com/voglinio/external-h-e-data-with-mask-annotations
    """
    if os.path.exists(annot_path) and os.path.exists(tissue_path) and os.path.exists(train_path):
        for annot_num, annotation_tif in tqdm(enumerate(os.listdir(tissue_path))):

            image_id = annotation_tif.split('.')[0]

            img = cv2.imread(os.path.join(tissue_path, annotation_tif))

            annotation_xml = image_id + '.xml'
            tree = minidom.parse(os.path.join(annot_path, annotation_xml))
            regions = tree.getElementsByTagName("Regions")[0].getElementsByTagName("Region")

            mask = np.zeros((img.shape[0], img.shape[1]))

            for mm, region in enumerate(regions):

                vertices = region.getElementsByTagName("Vertex")
                polygon = []
                for vertex in vertices:
                    x = float(vertex.attributes["X"].value)
                    y = float(vertex.attributes["Y"].value)
                    polygon.append([x, y])
                polygon = np.array(polygon)
                polygon = polygon.reshape((-1, 1, 2))
                polygon = np.round(polygon)
                polygon = polygon.astype(np.int32)
                polygon.shape
                cv2.fillPoly(mask, [polygon], mm+1)

            for mm, (img_piece, mask_piece) in enumerate(zip(split_image(img), split_image(mask))):

                path = train_path+image_id+'_'+str(mm)
                if not os.path.isdir(path):
                    os.mkdir(path)
                if not os.path.isdir(path+'/images/'):
                    os.mkdir(path+'/images/')

                if not os.path.isdir(path+'/masks/'):
                    os.mkdir(path+'/masks/')
                fname = path + '/images/' + image_id+'_'+str(mm) + '.png'
                cv2.imwrite(fname, img_piece)

                mask_piece = label(mask_piece)

                mask_piece = np.repeat(
                    mask_piece[:, :, np.newaxis], mask_piece.max(), axis=-1)
                mask_piece = np.equal(mask_piece, np.ones_like(
                    mask_piece)*np.arange(1, mask_piece.max()+1)).astype(np.uint8)

                fname = path+'/masks/'+image_id+'_'+str(mm)+'.h5'
                with h5py.File(fname, "w") as hf:
                    hf.create_dataset("arr", data=mask_piece)

    else:
        print('One of the paths provided does not exist')


if __name__ == '__main__':

    import time

    start = time.time()

    data_to_array()

    annot_path = '../data/external/annotations/'
    tissue_path = '../data/external/tissue_images/'
    train_path = '../data/stage1_train/'

    preprocess_external_data(annot_path, tissue_path, train_path)

    make_n_save_mosaic()



    print('Elapsed time', round((time.time() - start)/60, 1), 'minutes')
