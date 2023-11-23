# Neural network hyperparameters
WIDTH = 176
HEIGHT = 208
DEPTH = 1

# dataset constants
NUM_CLASSES = 4
HOSPITAL_SPLIT_FILE = 'hospital_split.json'
VAL_SPLIT = 0.3
PIN_BOOL = True

# Alzheimer constants
LABELS_ALZ = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# Tumor constants
LABELS_TUMOR = ["glioma", "meningioma", "notumor", "pituitary"]

# Train constants
NUM_ROUNDS = 3
NUM_EPOCHS = 1
BATCH_SIZE = 64

TIMEOUT_SECONDS = 600
EPSILON = 10 ** (-5)

TIMEOUT_DEVICES = 300 # pay attention to this

# similarity = ['single', 'multiple', 'averaged']
SIMILARITY = "single"

compile_info = {
    "optimizer": "Adam",
    "metrics": ["accuracy"],
}

RANDOM_SEED = 42

# path constants
HOSPITALS_FILE_PATH = (
    "./off_chain/hospitals.pkl"
)
# Alzheimer dataset
DATASET_TRAIN_PATH_ALZ = "./Database/train"
DATASET_TEST_PATH_ALZ = "./Database/test"
DATASET_LIMIT = None

# Tumor dataset
DATASET_TRAIN_PATH_TUM = "./Brain_Tumor/train"
DATASET_TEST_PATH_TUM = "./Brain_Tumor/test"

