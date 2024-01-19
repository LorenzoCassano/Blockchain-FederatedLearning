import os
import sys
import cv2
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from classHospital import Hospital
from sklearn.model_selection import train_test_split
import json
from constants import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_dataset(hospitals):
  hospital_dataset = {}
  for hospital_name in hospitals.keys():
    dataset_path = OFF_CHAIN + hospital_name  # Adjust the path
    hospital_dataset[hospital_name] = tf.data.Dataset.load(dataset_path)

  dataset_path = OFF_CHAIN + 'test'
  hospital_dataset['test'] = tf.data.Dataset.load(dataset_path)
  return hospital_dataset

def print_hospital_split(hospital_split):
    for key, value in hospital_split.items():
        print(f"Hospital {key} has {value:.2%} elements of the train dataset")


def generate_random_split(n, seed=RANDOM_SEED):
    # Generate n random numbers between 0 and 1
    random.seed(RANDOM_SEED)
    random_numbers = [random.uniform(0.1, 1) for _ in range(n)]

    # Normalize the numbers so their sum is equal to 1
    total = sum(random_numbers)
    normalized_numbers = [num / total for num in random_numbers]

    # Create the dictionary with keys as alpha, beta, gamma, etc.
    keys = [chr(ord('A') + i) for i in range(n)]
    result = dict(zip(keys, normalized_numbers))
    assert np.sum(normalized_numbers) == 1  # checking if all the elements are included
    return result


def set_reproducibility(seed=RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.keras.utils.set_random_seed(seed)


def preprocess_data(image, label):
    label = tf.one_hot(label, depth=4)
    return image, label


def createHospitals(train_path, test_path, hospital_split, dataset_name):
    labels = LABELS_ALZ if train_path == DATASET_TRAIN_PATH_ALZ else LABELS_TUMOR

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        seed=RANDOM_SEED,
        image_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_names=labels,
        color_mode='grayscale',
        shuffle=False
    )

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        seed=RANDOM_SEED,
        image_size=(WIDTH, HEIGHT),
        batch_size=None,  # it is needed for the split
        class_names=labels,
        color_mode='grayscale',
        shuffle=True
    )

    # one hot encoding

    train_dataset = train_dataset.map(preprocess_data)

    test_dataset = test_dataset.map(preprocess_data)

    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), reshuffle_each_iteration=False)
    # split the dataset for different devices
    hospitals = {}

    total_size = train_dataset.cardinality().numpy()

    split_sizes = {key: int(value * total_size) for key, value in
                   hospital_split.items()}  # dict for hospital e number of data for the hospital

    elements = split_sizes[list(split_sizes.keys())[-1]]  # number of data for last device

    split_sizes[list(split_sizes.keys())[-1]] = elements + (
                total_size - sum(split_sizes.values()))  # reset the number of data for last device

    hospital_dataset = {}
    samples_taken = 0
    for hospital_name, size in split_sizes.items():
        train_dataset_device = train_dataset.skip(samples_taken).take(size).batch(
            batch_size=32)  # Adjust batch_size as needed
        samples_taken += size

        hospitals[hospital_name] = Hospital(hospital_name, dataset_name)
        hospital_dataset[hospital_name] = train_dataset_device.shuffle(buffer_size = len(train_dataset_device),reshuffle_each_iteration=True)

    hospital_dataset['test'] = test_dataset
    return hospitals, hospital_dataset
def get_hospitals():
    hospitals = {}
    with open(HOSPITALS_FILE_PATH, "rb") as file:
        hospitals = pickle.load(file)
    return hospitals


def set_hospitals(hospitals):
    serialized_hospitals = pickle.dumps(hospitals)
    if not os.path.exists(OFF_CHAIN):
        os.makedirs(OFF_CHAIN)
    with open(HOSPITALS_FILE_PATH, "wb") as file:
        file.write(serialized_hospitals)

def save_dataset(hospital_dataset):
    if not os.path.exists(OFF_CHAIN):
        os.makedirs(OFF_CHAIN)

    for key, hospital in hospital_dataset.items():
        # set the Hospital file
        dataset_path = OFF_CHAIN + key
        # tf.data.save(hospital_obj.train_dataset, dataset_path)
        tf.data.Dataset.save(hospital, dataset_path)

def get_hospital_split():
    with open(HOSPITAL_SPLIT_FILE, 'r') as json_file:
        hospital_split = json.load(json_file)
    return hospital_split


def get_X_test():
    hospitals = get_hospitals()
    hospital_split = get_hospital_split()
    X_test = None
    try:
        X_test = np.concatenate(
            [
                hospitals[hospital_name].dataset["X_test"]
                for hospital_name in hospital_split
            ],
            axis=0,
        )
    except ValueError as e:
        print(
            "ERROR --> catch at:",
            __name__,
            "on file:",
            __file__,
            "\nException:",
            str(e),
        )
    if X_test is None:
        raise Exception(
            "ERROR --> catch at:",
            __name__,
            "on file:",
            __file__,
            "\nException: X_test is empty",
        )
    return X_test


def get_y_test():
    hospitals = get_hospitals()
    hospital_split = get_hospital_split()
    y_test = None
    try:
        y_test = np.concatenate(
            [
                hospitals[hospital_name].dataset["y_test"]
                for hospital_name in hospital_split
            ],
            axis=0,
        )
    except ValueError as e:
        print(
            "ERROR --> catch at:",
            __name__,
            "on file:",
            __file__,
            "\nException:",
            str(e),
        )
    if y_test is None:
        raise Exception(
            "ERROR --> catch at:",
            __name__,
            "on file:",
            __file__,
            "\nException: y_test is empty",
        )
    return y_test


def print_weights(weights):
    print(len(weights))
    print(type(weights))
    for w in weights:
        print(w.shape)
        print(type(w))
        print(str(sys.getsizeof(w)))
    print("weights size:" + str(sys.getsizeof(weights)))
    print(
        "weights TOTAL size:"
        + str(sys.getsizeof(weights) + sum(sys.getsizeof(w) for w in weights))
    )
    print_line("-")


def print_listed_weights(weights_listed):
    print(len(weights_listed))
    print(type(weights_listed))
    for w in weights_listed:
        print(len(w))
        print(type(w))
        print(str(sys.getsizeof(w)))
    print("weights_listed size:" + str(sys.getsizeof(weights_listed)))
    print(
        "weights_listed TOTAL size:"
        + str(
            sys.getsizeof(weights_listed)
            + sum(sys.getsizeof(w) for w in weights_listed)
        )
    )
    print_line("-")


def print_line(c):
    print(c * 50, "\n")


def device_out_of_battery(hospitals, n=1):
    devices = []
    hospitals_name = list(hospitals.keys())
    for _ in range(n):
        idx = random.randint(0, len(hospitals) - 1)
        # Different devices out of battery
        while hospitals_name[idx] in devices:
          idx = random.randint(0, len(hospitals) - 1)
        devices.append(hospitals_name[idx])
    return devices


def round_out_of_battery(rounds):
    return random.randint(1, rounds//2)  # not a round so far, for better sperimentation
