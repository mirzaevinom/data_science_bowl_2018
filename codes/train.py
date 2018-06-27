"""
Training part of my solution to The 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.

author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""

from config import *
import h5py


class KaggleDataset(utils.Dataset):
    """wrapper for loading bowl datasets
    """

    def load_shapes(self, id_list, train_path):
        """initialize the class with dataset info.
        """
        # Add classes
        self.add_class('images', 1, "nucleus")
        self.train_path = train_path

        # Add images
        for i, id_ in enumerate(id_list):
            self.add_image('images', image_id=i, path=None,
                           img_name=id_)

    def load_image(self, image_id, color):
        """Load image from directory
        """

        info = self.image_info[image_id]
        path = self.train_path + info['img_name'] + \
            '/images/' + info['img_name'] + '.png'

        img = load_img(path, color=color)

        return img

    def image_reference(self, image_id):
        """Return the images data of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'images':
            return info['images']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for images of the given image ID.
        """

        info = self.image_info[image_id]

        path = self.train_path + info['img_name'] + \
            '/masks/' + info['img_name'] + '.h5'
        if os.path.exists(path):
            # For faster data loading run augment_preprocess.py file first
            # That should save masks in a single h5 file
            with h5py.File(path, "r") as hf:
                mask = hf["arr"][()]
        else:
            path = self.train_path + info['img_name']
            mask = []
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                if 'png' in mask_file:
                    mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
                    mask_ = np.where(mask_ > 128, 1, 0)
                    # Fill holes in the mask
                    mask_ = binary_fill_holes(mask_).astype(np.int32)
                    # Add mask only if its area is larger than one pixel
                    if np.sum(mask_) >= 1:
                        mask.append(np.squeeze(mask_))

            mask = np.stack(mask, axis=-1)
            mask = mask.astype(np.uint8)

        # Class ids: all ones since all are foreground objects
        class_ids = np.ones(mask.shape[2])

        return mask.astype(np.uint8), class_ids.astype(np.int8)


def train_validation_split(train_path, seed=10, test_size=0.1):

    """
    Split the dataset into train and validation sets.
    External data and mosaics are directly appended to training set.
    """
    from sklearn.model_selection import train_test_split

    image_ids = list(
        filter(lambda x: ('mosaic' not in x) and ('TCGA' not in x), os.listdir(train_path)))
    mosaic_ids = list(filter(lambda x: 'mosaic' in x, os.listdir(train_path)))
    external_ids = list(filter(lambda x: 'TCGA' in x, os.listdir(train_path)))

    # Load and preprocess the dataset with train image modalities
    df = pd.read_csv('../data/classes.csv')
    df['labels'] = df['foreground'].astype(str) + df['background']
    df['filename'] = df['filename'].apply(lambda x: x[:-4])
    df = df.set_index('filename')
    df = df.loc[image_ids]

    # Split training set based on provided image modalities
    # This ensures that model validates on all image modalities.
    train_list, val_list = train_test_split(df.index, test_size=test_size,
                                            random_state=seed, stratify=df['labels'])

    # Add external data and mos ids to training list
    train_list = list(train_list) + mosaic_ids + external_ids
    val_list = list(val_list)

    return train_list, val_list


if __name__ == '__main__':
    import time


    train_path = '../data/stage1_train/'

    start = time.time()

    # Split the training set into training and validation
    train_list, val_list = train_validation_split(
        train_path, seed=11, test_size=0.1)

    # initialize training dataset
    dataset_train = KaggleDataset()
    dataset_train.load_shapes(train_list, train_path)
    dataset_train.prepare()

    # initialize validation dataset
    dataset_val = KaggleDataset()
    dataset_val.load_shapes(val_list, train_path)
    dataset_val.prepare()

    # Create model configuration in training mode
    config = KaggleBowlConfig()
    config.STEPS_PER_EPOCH = len(train_list)//config.BATCH_SIZE
    config.VALIDATION_STEPS = len(val_list)//config.BATCH_SIZE
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Model weights to start training with
    init_with = "imagenet"  # imagenet, last, or some pretrained model

    if init_with == "imagenet":
        weights_path = model.get_imagenet_weights()
        model.load_weights(weights_path, by_name=True)

    elif init_with == "last":
        # Load the last model you trained and continue training
        weights_path = model.find_last()[1]
        model.load_weights(weights_path, by_name=True)
    elif init_with == 'pretrained':
        weights_path = '../data/pretrained_model.h5'
        model.load_weights(weights_path, by_name=True)

    print('Loading weights from ', weights_path)

    # Train the model for 75 epochs

    model.train(dataset_train, dataset_val,
                learning_rate=1e-4,
                epochs=25,
                verbose=2,
                layers='all')

    model.train(dataset_train, dataset_val,
                learning_rate=1e-5,
                epochs=50,
                verbose=2,
                layers='all')

    model.train(dataset_train, dataset_val,
                learning_rate=1e-6,
                epochs=75,
                verbose=2,
                layers='all')

    print('Elapsed time', round((time.time() - start)/60, 1), 'minutes')
