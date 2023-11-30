# Neural network hyperparameters
WIDTH = 176
HEIGHT = 208
DEPTH = 1

#Dataset name
BRAIN_TUMOR = "Brain_Tumor"
ALZHEIMER = "Alzheimer"

# dataset constants
NUM_CLASSES = 4
HOSPITAL_SPLIT_FILE = 'hospital_split.json'
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
PIN_BOOL = True

# Alzheimer constants
LABELS_ALZ = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# Tumor constants
LABELS_TUMOR = ["glioma", "meningioma", "notumor", "pituitary"]

# Train constants
NUM_ROUNDS = 4
NUM_EPOCHS = 5
BATCH_SIZE = 32

TIMEOUT_SECONDS = 800
EPSILON = 10 ** (-5)

TIMEOUT_DEVICES = 800 # pay attention to this

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

