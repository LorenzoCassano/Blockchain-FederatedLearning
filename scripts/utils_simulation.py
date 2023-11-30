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


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    if DATASET_LIMIT:
        for dir1 in os.listdir(img_folder):
            for idx, file in enumerate(os.listdir(os.path.join(img_folder, dir1))):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, 0)
                image = np.array(image)
                image = image.astype("float32")
                img_data_array.append(image)
                class_name.append(dir1)

                if idx == DATASET_LIMIT:
                    break
    else:
        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, 0)
                image = np.array(image)
                image = image.astype("float32")
                img_data_array.append(image)
                class_name.append(dir1)
    return img_data_array, class_name


def createHospitals(train_path, test_path, hospital_split, dataset_name):
    hospitals = {}

    # extract the image array and class name
    img_data, class_name = create_dataset(train_path)
    img_data_test, class_name_test = create_dataset(test_path)

    """
    target_dict = {
        "NonDemented": 0,
        "VeryMildDemented": 1,
        "MildDemented": 2,
        "ModerateDemented": 3,
    }
    """
    labels = LABELS_ALZ if train_path == DATASET_TRAIN_PATH_ALZ else LABELS_TUMOR

    target_dict = {label: index for index, label in enumerate(labels)}

    target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

    target_val_test = [target_dict[class_name_test[i]] for i in range(len(class_name_test))]

    X = np.array(img_data, np.float32)
    y = np.array(list(map(int, target_val)), np.float32)

    X_test = np.array(img_data_test, np.float32)
    y_test = np.array(list(map(int, target_val_test)), np.float32)

    rows = len(X)
    values_list = []
    for hospital_name in hospital_split:
        values_list += [hospital_name] * int(rows * hospital_split[hospital_name])

    if len(values_list) < len(X):  # case useful for approximation
        difference = len(X) - len(values_list)
        values_list += [list(hospital_split.keys())[-1]] * difference  # adding last element

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    df = pd.DataFrame({"X": list(X), "y": list(y)})
    if df.shape[0] != len(values_list):
        values_list.append("Gamma")

    df["hospital"] = values_list
    # df['hospital'] = df['hospital'].map(hospitals)

    dataset = dict.fromkeys(list(hospital_split.keys()))

    for hospital_name in hospital_split:
        X_h = df[df["hospital"] == hospital_name]["X"].to_numpy()
        y_h = df[df["hospital"] == hospital_name]["y"].to_numpy()

        X_h = np.stack(X_h, axis=0)
        y_h = np.stack(y_h, axis=0)

        dataset[hospital_name] = {}

        if train_path == DATASET_TRAIN_PATH_ALZ:
            (X_train, X_test, y_train, y_test,) = train_test_split(X_h, y_h, test_size=VAL_SPLIT,
                                                                   random_state=RANDOM_SEED)
            (
                X_test,
                X_val,
                y_test,
                y_val,
            ) = train_test_split(X_test, y_test, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
        else:
            (X_train, X_val, y_train, y_val,) = train_test_split(X_h, y_h, test_size=VAL_SPLIT,
                                                                 random_state=RANDOM_SEED)

        dataset[hospital_name]["X_train"] = np.expand_dims(np.array(X_train, np.float32), axis=-1)
        dataset[hospital_name]["y_train"] = tf.one_hot(y_train, depth=len(labels))
        dataset[hospital_name]["X_test"] = np.expand_dims(np.array(X_test, np.float32), axis=-1)
        dataset[hospital_name]["y_test"] = tf.one_hot(y_test, depth=len(labels))
        dataset[hospital_name]["X_val"] = np.expand_dims(np.array(X_val, np.float32), axis=-1)
        dataset[hospital_name]["y_val"] = tf.one_hot(y_val, depth=len(labels))

        hospitals[hospital_name] = Hospital(hospital_name, dataset[hospital_name], dataset_name)


    return hospitals


def get_hospitals():
    hospitals = {}
    with open(HOSPITALS_FILE_PATH, "rb") as file:
        hospitals = pickle.load(file)
    return hospitals


def set_hospitals(hospitals):
    serialized_hospitals = pickle.dumps(hospitals)
    with open(HOSPITALS_FILE_PATH, "wb") as file:
        file.write(serialized_hospitals)


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
        devices.append(hospitals_name[idx])
    return devices


def round_out_of_battery(rounds):
    return random.randint(1, rounds - 1)  # device cannot be out of memeory at first round
