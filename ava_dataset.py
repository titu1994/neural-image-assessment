from path import Path
from config import dataset_path, base_images_path
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize


def get_image_paths_and_scores(txt_file, base_images_path):
    """ reads a file in the structure of AVA.txt, normalizes the scores
        and returns the list of files and the scores as a numpy array.
    :param txt_file: contains shuffled lines from AVA.txt for
                     training or testing dataset
    :param base_images_path: the directory containing the AVA images
    :return: (image_paths, normalized scores)
    """
    image_paths = []
    scores = []
    path_template = base_images_path.joinpath("{}" + '.jpg')
    with open(txt_file, mode='r') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines)):
            token = line.split()
            id = int(token[1])
            values = np.array(token[2:12], dtype='float32')
            file_path = path_template.format(id)
            image_paths.append(file_path)
            scores.append(values)
    scores = normalize(np.array(scores, dtype='float32'), axis=1, norm='l1')
    return image_paths, scores


class AvaDataset(object):
    """ contains paths and scores data for the training and test set
    """
    def __init__(self, dataset_path, base_images_path, max_train=0, max_test=0):
        """ load paths and scores
        :param dataset_path: folder containing the files train_files.txt and
                             test_files.txt
        :param base_images_path: the folder containing the jpg images of the
                                 AVA dataset
        :param max_train: if != 0, limits how many images are used for training
        :param max_test: if != 0, limits how many images are used for testing
        """
        print("Loading training set and val set")
        self.train_image_paths, self.train_scores = get_image_paths_and_scores(
            dataset_path.joinpath('train_files.txt'),
            base_images_path)
        self.test_image_paths, self.test_scores = get_image_paths_and_scores(
            dataset_path.joinpath('test_files.txt'),
            base_images_path)

        if max_train != 0:
            self.train_image_paths = self.train_image_paths[:max_train]
            self.train_scores = self.train_scores[:max_train]
        if max_test != 0:
            self.test_image_paths = self.test_image_paths[:max_test]
            self.test_scores = self.test_scores[:max_test]

        self.test_size = len(self.test_image_paths)
        self.total_image_count = self.test_size + len(self.train_image_paths)


#dataset = Dataset(dataset_path=dataset_path, base_images_path=base_images_path)