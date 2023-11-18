# Neural network hyperparameters
WIDTH = 176
HEIGHT = 208
DEPTH = 1

# dataset constants
NUM_CLASSES = 4
LABELS = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
HOSPITAL_SPLIT = {"Alpha": 0.5, "Beta": 0.3, "Gamma": 0.2}
VAL_SPLIT = 0.3
PIN_BOOL = True

# Train constants
NUM_ROUNDS = 3
NUM_EPOCHS = 1
BATCH_SIZE = 64

TIMEOUT_SECONDS = 600
EPSILON = 10 ** (-5)

TIMEOUT_DEVICES = 180

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
DATASET_TRAIN_PATH = "./Database/train"
DATASET_TEST_PATH = "./Database/test"
DATASET_LIMIT = None


