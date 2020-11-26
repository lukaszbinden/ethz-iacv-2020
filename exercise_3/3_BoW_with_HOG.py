"""
Bag-of-Words classification of STL-10 dataset with HOG features and nearest-neighbor classifier.
Usage:
python 3_BoW_with_HOG.py --dataset /home/cvcourse/pics/STL-10/images_per_class --classes cat ship
"""


import argparse
import sys
import os
import glob
import cv2
import numpy as np
from scipy import ndimage, spatial


# Helper functions.

def parse_args():
    """
    Parse input arguments.
    """

    parser = argparse.ArgumentParser(description='Bag-of-Words classification with HOG features.')
    parser.add_argument('--dataset', dest='dataset_dir', required=True, help='Path to the root dataset directory.')
    parser.add_argument('--classes', dest='class_names', nargs='+', default=['cat', 'ship'],
                        help='List of class names. Default: cat, ship.')
    parser.add_argument('--clusters', dest='K', type=int, default=100,
                        help='Number of codebook words, i.e. K-means clusters.')
    parser.add_argument('--nn_norm', dest='nearest_neighbor_norm', default='L2', choices=['L2'],
                        help='Norm used to find nearest neighbor of BoW histograms.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def grid_of_feature_points(image, n_points_x, n_points_y, margin_x, margin_y):
    """
    Construct grid of feature points to serve as patch centers for computation of HOG features.
    """

    # Return the set of feature points as two 1D arrays holding their image coordinates.
    return feature_points_x, feature_points_y


def compute_HOG_descriptors(image, feature_points_x, feature_points_y, cell_width, cell_height):
    """
    Compute the HOG descriptors, as the set of features for an input image, at the specified points.
    Output:
        |HOG_descriptors|: 2D NumPy array of shape (n_points, n_cells * n_cells * n_bins)
    """

    # Define parameters and constants.
    n_bins = 8
    n_points = feature_points_x.shape[0]
    n_cells = 4
    pi = np.pi

    return HOG_descriptors


def feature_extraction(image_full_filename):
    """
    Extract HOG features for an input image.
    Inputs:
        |image_full_filename|: full path to the input image file
    Output:
        2D NumPy array of shape (n_points_x * n_points_y, 128)
    """

    # Read the input image into a numpy.ndarray variable of two dimensions (grayscale) for further processing.
    image = cv2.imread(image_full_filename, 0).astype('float')

    # Define parameters.
    n_points_x = 6
    n_points_y = 6
    cell_width = 4
    cell_height = 4
    margin_x = 2 * cell_width
    margin_y = 2 * cell_height

    # Construct grid of feature points.
    feature_points_x, feature_points_y = grid_of_feature_points(image, n_points_x, n_points_y, margin_x, margin_y)

    # Return HOG features at the computed feature points.
    return compute_HOG_descriptors(image, feature_points_x, feature_points_y, cell_width, cell_height)


def image_full_filenames_in_directory(directory):
    """
    Return a list with full filenames of all images in the input directory, sorted in lexicographical order.
    Inputs:
        |directory|: path to input directory.
    """

    image_format = '.png'
    image_filename_pattern = os.path.join(directory, '*' + image_format)
    list_image_full_filenames = glob.glob(image_filename_pattern)
    # Sort the list.
    list_image_full_filenames = sorted(list_image_full_filenames)

    return list_image_full_filenames


def class_features(class_directory):
    """
    Construct a 3D numpy.ndarray holding the HOG features for all images in a class, under the input directory.
    Inputs:
        |class_directory|: path to input directory.
    """

    # Get the list with all images in the class directory.
    list_image_full_filenames = image_full_filenames_in_directory(class_directory)
    n_images = len(list_image_full_filenames)

    # Initialize a list of HOG features per image.
    HOG_features = []

    # Main loop over the images to compute and append HOG features.
    for i in range(n_images):
        # Display progress.
        print('Feature extraction for image {:d}/{:d}'.format(i + 1, n_images))

        # Extract features for current image as a 2D numpy.ndarray and append it to the list.
        HOG_features.append(feature_extraction(list_image_full_filenames[i]))

    # Concatenate feature vectors from all images into a single 3D numpy.ndarray with dimensions
    # n_images-by-n_descriptors-by-D.
    # ASSUMPTION: all images of processed classes have equal dimensions, therefore equal n_points for the constructed
    # grids.
    HOG_features_class = np.array(HOG_features)

    return HOG_features_class


def split_features(dataset_dir, split, class_names):
    """
    Construct a list of 3D arrays, one for each class, with features for an entire split of the dataset.
    Inputs:
        |dataset_dir|: path to root dataset directory.
        |split|: name of processed split, e.g. 'train' or 'test'.
        |class_names|: list of names of considered classes.
    """

    # Form path to root split directory.
    split_dir = os.path.join(dataset_dir, split)

    HOG_features_split = []

    # Main loop over classes.
    for i in range(len(class_names)):
        current_class_name = class_names[i]

        # Display progress.
        print('Processing {:s} split, class {:d}: {:s}'.format(split, i + 1, current_class_name))

        # Extract features.
        HOG_features_split.append(class_features(os.path.join(split_dir, current_class_name)))

    return HOG_features_split


def find_nearest_neighbor_L2(points_1, points_2):
    """
    Determine the nearest neighbor of each point of the first set from the second set in the L2-norm sense.
    Inputs:
        |points_1|: 2D numpy.ndarray containing the first set of points, with dimensions N-by-D.
        |points_2|: 2D numpy.ndarray containing the second set of points, with dimensions K-by-D.
    Output:
        1D NumPy array with N elements, corresponding to the indices of points in |points_2| that are the nearest
        neighbors of points in |points_1|
    """

    return nearest_neighbor_indices


def kmeans(points, K, n_iter):
    """
    Cluster the input points into K clusters using K-means with the specified number of iterations and output the
    induced cluster centroids.
    Inputs:
        |points|: 2D numpy.ndarray containing feature vectors as its rows, with dimensions N-by-D
        |K|: number of clusters
        |n_iter|: number of iterations of K-means algorithm
    Output:
        |centroids|: 2D numpy.ndarray containing the final cluster centroids as its rows, with dimensions K-by-D
    """

    N, n_dims = points.shape[:2]

    # Centroid initialization with randomly selected feature vectors.
    # centroids = ...

    # Main K-means loop.
    for i in range(n_iter):
        # 1) Cluster assignment.

        # 2) Centroid update based on current assignment.
        for k in range(K):
            # Check if cluster is empty.

        # Display progress.
        print('Completed K-means iteration {:d}/{:d}'.format(i+1, n_iter))

    return centroids


def bow_histograms_and_labels(HOG_features_split, codebook_words):
    """
    Compute the Bag-of-Words histograms for an entire split of the dataset, using the respective codebook with visual
    words that has been computed with K-means. Also create an array of ground truth labels for images in the split.
    Inputs:
        |HOG_features_split|: list of 3D arrays, one for each class, in which each array holds the features for all
        images in the split that belong to that class
        |codebook_words|: 2D numpy.ndarray containing codebook words as its rows, with dimensions K-by-D
    """

    C = len(HOG_features_split)
    K, D = codebook_words.shape

    # Initialize matrix of BoW histograms and array of ground truth labels.
    bow_histograms_split = np.empty((0, K))
    labels_split = np.empty((0, 1), dtype=int)

    for c in range(C):
        HOG_features_class = HOG_features_split[c]
        n_images = HOG_features_class.shape[0]

        # Add labels of current class to overall label array.
        labels_split = np.concatenate((labels_split, c + np.zeros((n_images, 1), dtype=int)))

        # Initializations.
        bow_histograms_class = np.zeros((n_images, K))

        # Loop over all images in the class and compute BoW histograms.
        for i in range(n_images):
            # |HOG_features_image| is a 2D numpy.ndarray containing all HOG descriptors of the current image as its rows.
            HOG_features_image = HOG_features_class[i]
            # Assign each descriptor of the current image to a word.
            # ...
            # Count how many descriptors are assigned to each word.
            # bow_histograms_class[i, :] = ...

        # Append BoW histograms for images in current class to the overall split-level matrix.
        bow_histograms_split = np.concatenate((bow_histograms_split, bow_histograms_class))

    return bow_histograms_split, labels_split


def nearest_neighbor_classifier(points_test, points_train, labels_train, norm='L2'):
    """
    Classify test points by assigning to each of them the label of its nearest neighbor point from the training set.
    Inputs:
        |points_test|: 2D numpy.ndarray containing the test points as its rows, with dimensions S-by-K.
        |points_train|: 2D numpy.ndarray containing the train points as its rows, with dimensions T-by-K.
        |labels_train|: 1D numpy.ndarray containing the ground truth labels of the train points, with dimensions T-by-1.
    """

    # Compute nearest neighbors.
    if norm == 'L2':
        # ...
    else:
        # ...

    # Assign to test points the label of their nearest training neighbor.
    # labels_test = ...

    return labels_test


def confusion_matrix(labels_ground_truth, labels_predicted, C):
    """
    Compute the confusion matrix based on the ground truth labels and the respective predictions.
    Inputs:
        |labels_ground_truth|: 1D numpy.ndarray containing the ground truth labels, with dimensions S-by-1.
        |labels_predicted|: 1D numpy.ndarray containing the predicted labels, with same dimensions as
                            |labels_ground_truth|.
    """

    # Initialize confusion matrix to zero values.
    conf = np.zeros((C, C))

    # Use definition of confusion matrix to compute its values: rows correspond to ground truth labels, columns to
    # predictions.
    np.add.at(conf, (labels_ground_truth, labels_predicted), 1)

    return conf


def accuracy_from_confusion_matrix(conf):
    """
    Compute the accuracy of a classifier from the confusion matrix related to its predictions.
    Input:
        |conf|: confusion matrix as a 2D numpy.ndarray, with dimensions C-by-C.
    """

    accuracy = np.trace(conf) / np.sum(conf)

    return accuracy

# ----------------------------------------------------------------------------------------------------------------------

# Main function.
if __name__ == '__main__':

    # Parse input arguments.
    args = parse_args()
    dataset_dir = args.dataset_dir
    class_names_input = args.class_names
    K = args.K
    nearest_neighbor_norm = args.nearest_neighbor_norm

    # Filter class names to obtain a subset of STL-10 classes. If this subset has less than two elements, exit with an
    # error status.
    STL10_class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    class_names_input_unique = np.unique(class_names_input)
    is_input_valid = np.array([c in STL10_class_names for c in class_names_input_unique])
    class_names = class_names_input_unique[is_input_valid]
    C = len(class_names)
    if C < 2:
        print('Not enough classes to distinguish. Need at least 2 classes from STL-10!')
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------

    # TRAINING - STEP 1) Compute HOG features for the entire train split.

    train_split = 'train'
    HOG_features_train = split_features(dataset_dir, train_split, class_names)

    # Concatenate HOG features from all classes of the train split into one 2D matrix.
    n_images_per_class, n_descriptors_per_image, D = HOG_features_train[0].shape
    HOG_features_train_concatenated = np.empty((0, D))
    for c in range(C):
        HOG_features_train_concatenated = np.concatenate((HOG_features_train_concatenated,
                                                          np.reshape(HOG_features_train[c], (-1, D))))

    # ------------------------------------------------------------------------------------------------------------------

    # TESTING - STEP 1) Compute HOG features for the entire test split.

    test_split = 'test'
    HOG_features_test = split_features(dataset_dir, test_split, class_names)

    # ------------------------------------------------------------------------------------------------------------------

    # TRAINING + TESTING - STEP 2)

    n_evaluation_rounds = 10

    # Initialize confusion matrices and array of accuracy values.
    confusion_matrices = np.zeros((n_evaluation_rounds, C, C))
    accuracy_values = np.zeros(n_evaluation_rounds)

    # Fix random seed to ensure reproducibility of the results.
    np.random.seed(0)

    # Define other parameters.
    n_iters_kmeans = 10

    # Main loop to repeat training and testing.
    for i in range(n_evaluation_rounds):

        print('Running {:d}/{:d} evaluation round for Bag-of-Words classification'.format(i+1, n_evaluation_rounds))

        # TRAINING - STEP 2)i) Construct the codebook of HOG feature vectors by applying K-means to the entire set of
        # training features.
        print('Constructing codebook from training features using K-means...')
        codebook_words = kmeans(HOG_features_train_concatenated, K, n_iters_kmeans)
        print('Codebook constructed.')

        # TRAINING - STEP 2)ii) Compute the Bag-of-Words histogram representation of all training images that is induced
        # by the constructed codebook.
        bow_histograms_train, labels_train = bow_histograms_and_labels(HOG_features_train, codebook_words)

        # TESTING - STEP 2)i) Compute the Bag-of-Words histogram representation of all testing images that is induced
        # by the constructed codebook.
        bow_histograms_test, labels_test_ground_truth = bow_histograms_and_labels(HOG_features_test, codebook_words)

        # TESTING - STEP 2)ii) Predict test labels with nearest-neighbor classifier.
        labels_test_predicted = nearest_neighbor_classifier(bow_histograms_test, bow_histograms_train, labels_train,
                                                            nearest_neighbor_norm)

        # TESTING - STEP 2)iii) Evaluate the predictions of the classifier on the test split against ground truth.
        confusion_matrices[i] = confusion_matrix(labels_test_ground_truth, labels_test_predicted, C)
        accuracy_values[i] = accuracy_from_confusion_matrix(confusion_matrices[i])

    # Report cumulative results over all evaluation rounds.
    accuracy_average = np.mean(accuracy_values)
    accuracy_std = np.std(accuracy_values, ddof=1)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
    print('Average BoW classification accuracy over {:d} rounds: {:6.2f}% +/- {:5.2f}%'.format(n_evaluation_rounds,
                                                                                               100 * accuracy_average,
                                                                                               100 * (3 * accuracy_std)))























