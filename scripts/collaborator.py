import os
import sys
from numpy import require

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils_simulation import get_hospitals, print_line, set_reproducibility, round_out_of_battery, device_out_of_battery
from utils_collaborator import *
from brownie import FederatedLearning
from classHospital import Hospital
import ipfshttpclient
from tensorflow.keras.models import model_from_json
import asyncio
import json
import pickle
from constants import NUM_ROUNDS

from fedAvg import FedAvg
from fedProx import FedProx

from constants import *
import random
import time

set_reproducibility()

# retrieve the hospitals information
hospitals = get_hospitals()
# connect to IPFS and Blockchain
IPFS_client = ipfshttpclient.connect()

FL_contract = FederatedLearning[-1]

# manage contract events
contract_events = FL_contract.events

# storing the hospitals performance results through the Federated Learning rounds
hospitals_evaluation = {hospital_name: [] for hospital_name in hospitals}

ROUND_BATTERY = 100  # setting an higher time to avoid out of battery
DEVICES_OUT_OF_BATTERY = []
file_name = ""
if "out" in sys.argv:
    ROUND_BATTERY = round_out_of_battery(NUM_ROUNDS)
    DEVICES_OUT_OF_BATTERY = device_out_of_battery(hospitals, n=NUM_DEVICES_OUT_BATTERY)
    print(f"Device/s {DEVICES_OUT_OF_BATTERY} will be out of battery at round {ROUND_BATTERY + 1}")
    name = f"{NUM_ROUNDS}_{NUM_EPOCHS}_{DEVICES_OUT_OF_BATTERY}"
else:
    print("No devices out of battery")
    name = f"{NUM_ROUNDS}_{NUM_EPOCHS}_{[]}"

with open('devices_out_of_battery.pkl', 'wb') as file:
    # Use pickle.dump to save the list to the file
    pickle.dump(DEVICES_OUT_OF_BATTERY, file)



def closeState_alert(event):
    print("The FL Blockchain has been CLOSED\n")
    print("RESULTS - Hospitals Performance Evaluation through Federated Learning...")
    for hospital_name in hospitals_evaluation:
        print(f"{hospital_name}:")
        for round, [loss, acc] in enumerate(hospitals_evaluation[hospital_name], start=1):
            print(f"\tRound {round}:\tLoss: {loss:.3f} - Accuracy: {acc:.3f}")
    # network.disconnect()
    # sys.exit(0)


# triggered after the START event from the Blockchain
def start_event():
    for hospital_name in hospitals:
        # retrieving of the model given by the Manager
        retrieve_model_tx = FL_contract.retrieve_model(
            {"from": hospitals[hospital_name].address}
        )
        retrieve_model_tx.wait(1)
        custom_objects = {'FedAvg': FedAvg, 'FedProx': FedProx}
        decoded_model = decode_utf8(retrieve_model_tx)
        print("decoded_model: ", decoded_model)
        model = model_from_json(decoded_model, custom_objects=custom_objects)
        print("Model ", model)
        hospitals[hospital_name].model = model

        # retrieving of the compile information goven by the Manager
        retreive_compile_info_tx = FL_contract.retrieve_compile_info(
            {"from": hospitals[hospital_name].address}
        )
        retreive_compile_info_tx.wait(1)

        decoded_compile_info = decode_utf8(retreive_compile_info_tx)
        fl_compile_info = json.loads(decoded_compile_info)
        hospitals[hospital_name].compile_info = fl_compile_info

        # compiling the model with the compile information
        hospitals[hospital_name].model.compile(**hospitals[hospital_name].compile_info)


# operations to do at every FL round
def round_loop(round, fed_dict, file_name):
    for hospital_name in hospitals:
        if hospital_name not in fed_dict:
            fed_dict[hospital_name] = {}
        if round >= ROUND_BATTERY and hospital_name in DEVICES_OUT_OF_BATTERY:
            print(f"Device {hospital_name} is out of battery")
            fed_dict[hospital_name][round] = "out_of_battery"
        else:
            print(f"Device {hospital_name} is training ...")
            fed_dict = fitting_model_and_loading_weights(hospital_name, round, fed_dict)
    dir_path = file_name.split('_')[0]
    if dir_path == 'Brain':
        dir_path = 'Brain_Tumor'
    path = './results/' + dir_path +'/'+ file_name + '.json'
    with open(path, 'w') as json_file:
        json.dump(fed_dict, json_file)
    return fed_dict


# triggered after the 'aggregatedWeightsReady' event from the Blockchain
def aggregatedWeightsReady_event():
    for hospital_name in hospitals:
        retrieving_aggreagted_weights(hospital_name)


def find_data(_hospital_name):
    X_train = hospitals[_hospital_name].dataset["X_train"]
    y_train = hospitals[_hospital_name].dataset["y_train"]
    X_val = hospitals[_hospital_name].dataset["X_val"]
    y_val = hospitals[_hospital_name].dataset["y_val"]
    X_test = hospitals[_hospital_name].dataset["X_test"]
    y_test = hospitals[_hospital_name].dataset["y_test"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def fitting_model_and_loading_weights(_hospital_name, round, fed_dict):
    """fitting the model"""

    X_train, y_train, X_val, y_val, X_test, y_test = find_data(_hospital_name)
    # Random epoch for FedProx
    epochs = random.randint(1, NUM_EPOCHS) if isinstance(hospitals[_hospital_name].model, FedProx) else NUM_EPOCHS
    print(f"Number of epoch for {_hospital_name} is {epochs}")
    fed_dict[_hospital_name][round] = {}
    for epoch in range(epochs):
        # Training phase
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_x = X_train[i:i + BATCH_SIZE]
            batch_y = y_train[i:i + BATCH_SIZE]

            train_loss = hospitals[_hospital_name].model.train_step(batch_x, batch_y)

        # Validation phase
        for i in range(0, len(X_val), BATCH_SIZE):
            val_batch_x = X_val[i:i + BATCH_SIZE]
            val_batch_y = y_val[i:i + BATCH_SIZE]

            val_loss = hospitals[_hospital_name].model.val_step(val_batch_x, val_batch_y)

        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch + 1}, Training Loss={mean_train_loss:.4f}, Validation Loss={mean_val_loss:.4f}")
        fed_dict[_hospital_name][round][epoch] = [str(mean_train_loss), str(mean_val_loss)]
    hospitals_evaluation[_hospital_name].append(
        hospitals[_hospital_name].model.evaluate(X_test, y_test)
    )

    # updating the old weights with the parameters from the newly fitted model
    hospitals[_hospital_name].weights = hospitals[_hospital_name].model.get_weights()

    """ loading weights """
    weights = hospitals[_hospital_name].weights
    weights_bytes = weights_encoding(weights)
    # print("weights_JSON size:" + str(sys.getsizeof(weights_JSON)))

    # uploading the weights on IPFS
    start_time = time.time()
    add_info = IPFS_client.add(weights_bytes, pin=PIN_BOOL)
    print("IPFS 'add' time: ", str(time.time() - start_time))
    print("IPFS 'add' info: ", add_info.keys())

    # sending the IPFS hash of the weights in the Blockchain
    hash_encoded = add_info["Hash"].encode("utf-8")
    send_weights_tx = FL_contract.send_weights(
        hash_encoded,
        {"from": hospitals[_hospital_name].address},
    )
    send_weights_tx.wait(1)
    return fed_dict


def retrieving_aggreagted_weights(_hospital_name):
    # retrieve the IPFS hash of the aggregated wights from the Blockchain
    retrieve_aggregated_weights_tx = FL_contract.retrieve_aggregated_weights(
        {"from": hospitals[_hospital_name].address}
    )
    retrieve_aggregated_weights_tx.wait(1)

    weight_hash = decode_utf8(retrieve_aggregated_weights_tx)

    # download the aggregated weights from IPFS
    start_time = time.time()
    aggregated_weights_encoded = IPFS_client.cat(weight_hash)
    print("IPFS 'cat' time: ", str(time.time() - start_time))

    aggregated_weights = weights_decoding(aggregated_weights_encoded)

    # setting the model with the new aggregated weights computed by the Manager
    hospitals[_hospital_name].aggregated_weights = aggregated_weights
    if isinstance(hospitals[_hospital_name].model, FedProx):
        print("Restore weights setting aggregator_weights: FEDPROX")
        FedProx.SERVER_WEIGHTS = aggregated_weights
    else:  # FedAvg
        print("Restore weights setting the weights of aggregator: FEDAVG")
        hospitals[_hospital_name].model.set_weights(aggregated_weights)


async def main():
    # continuously listens for the CLOSE event from the Blockchain and promptly handles it
    contract_events.subscribe("CloseState", closeState_alert, delay=0.5)
    print("Subscribed to CLOSE...")

    # await for the START event
    coroutine_start = contract_events.listen("StartState")
    print("COROUTINE: waiting START...\n", coroutine_start)
    await coroutine_start
    print("I waited START")
    print_line("_")
    # continue after reception
    start_event()

    # await for the LEARNING event
    coroutine_learning = contract_events.listen("LearningState")
    print("COROUTINE: waiting LEARNING...\n", coroutine_learning)
    await coroutine_learning
    print("I waited LEARNING")
    print_line("_")

    # Initialiazation weights
    hospital_name = list(hospitals.keys())[0]  # take the first elements to check the model used
    if isinstance(hospitals[hospital_name].model, FedProx):
        print("FedProx model weights initialization...")
        global_model = FedAvg(num_classes=4)
        global_model.build((None, WIDTH, HEIGHT, DEPTH))
        global_model.compile(optimizer="adam", metrics="accuracy")
        weights = global_model.trainable_weights
        assert len(weights) != 0
        FedProx.SERVER_WEIGHTS = weights
        file = f'FedProx{MU}_' + name
    else:
        file = 'FedAvg_' + name

    dataset = hospitals[hospital_name].dataset_name
    file_name = dataset + '_' + file
    print('new file ',file_name)


    # start of the Federated Learning loop that will be ended by the CLOSE event
    # set only for out of memory: debug
    round = 0
    fed_dict = {}
    while True:
        print("Start round loop ...")
        fed_dict = round_loop(round, fed_dict, file_name)
        print("fed_dict: ", fed_dict)
        # await for the Manager to send the aggregated weights
        coroutine_AW = contract_events.listen("AggregatedWeightsReady")
        print("COROUTINE: waiting 'send_aggreagted_weights'...\n", coroutine_AW)
        await coroutine_AW
        print("Aggregated weights arrived")
        print_line("_")
        # continue after reception
        aggregatedWeightsReady_event()
        round += 1


asyncio.run(main())
