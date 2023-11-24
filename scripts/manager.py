import os
import sys

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils_simulation import get_X_test, get_y_test, print_line, set_reproducibility
from utils_manager import *

from brownie import FederatedLearning, network, accounts
from deploy_FL import get_account
import ipfshttpclient

from constants import *
from sklearn.metrics import classification_report
import numpy as np
import asyncio
import time


set_reproducibility()

# connect to IPFS and Blockchain
IPFS_client = ipfshttpclient.connect()
FL_contract = FederatedLearning[-1]
manager = get_account()

# manage contract events
contract_events = FL_contract.events

# storing the overall performance results through the Federated Learning rounds
FL_evaluation = []
FL_classification_report = []

"""
# for SIMULATION/EVALUATION purposes:   - Observe FL model evaluation performance only on SIMULATION env
#                                       - On PRODUCTION env the Manager cannot afford sensitive (test) data from the collaborators
"""

"""
IMPORTANT:
Parameter setting
"""
model_test = FedAvg(NUM_CLASSES) # creation of the global model always FedAvg, only useful to store weights
model_test.compile(**compile_info)
model_test.build((None, WIDTH, HEIGHT, DEPTH))
X_test = get_X_test()
y_test = get_y_test()

model_used = "FedAvg" # model used by collaborators

if "FedProx" in sys.argv:
    model_used = "FedProx"

def retrive_information():
    # retrieving the parameters IPFS hashes loaded by the collaborators
    hospitals_addresses = FL_contract.get_collaborators({"from": manager})
    retrieved_weights_hash = {}
    for hospital_address in hospitals_addresses:
        retrieved_weights_hash[hospital_address] = FL_contract.retrieve_weights(
            hospital_address, {"from": manager}
        )
    hospitals_weights = {}

    # retrieving the collaborators weights from IPFS
    for hospital_address in retrieved_weights_hash:
        weights_hash = retrieved_weights_hash[hospital_address].decode("utf-8")
        if weights_hash == "":
            print(f"I did not receive anything by: {hospital_address}")
        else:
            start_time = time.time()
            weights_encoded = IPFS_client.cat(weights_hash)
            print("IPFS 'cat' time:", str(time.time() - start_time))
            weights = weights_decoding(weights_encoded)
            hospitals_weights[hospital_address] = weights # inserting weights only for whom send the weights
        # print_listed_weights(hospitals_weights[hospital_address])

    hospitals_number = len(hospitals_weights)
    weights_dim = len(hospitals_weights[list(hospitals_weights.keys())[0]])
    return weights_dim, hospitals_weights,hospitals_number, hospitals_addresses

def test_information(aggregated_weights):
    """
        function to obtain information about the global model
    """
    model_test.set_weights(aggregated_weights)
    results = model_test.predict(X_test)
    y_predicted = list(map(np.argmax, results))
    labels_y_test = np.argmax(y_test, axis=1)
    FL_classification_report.append(
        classification_report(
            labels_y_test,
            y_predicted,  # labels=LABELS
        )
    )
    #print("y_predicted: ", y_predicted)
    #print("y_test: ", labels_y_test)
    FL_evaluation.append(model_test.evaluate(X_test, y_test))

def federated_learning():

    weights_dim, hospitals_weights, hospitals_number, hospitals_addresses = retrive_information()
    # computing the AVERAGE of the collaborators weights
    averaged_weights = []

    for i in range(weights_dim):
        layer_weights = []
        for hospital_address in hospitals_weights:
            layer_weights.append(hospitals_weights[hospital_address][i])
        averaged_weights.append(sum(layer_weights) / hospitals_number)

    for i in range(len(averaged_weights)):
        averaged_weights[i] = np.array(
            averaged_weights[i]
        )  # convert the list to a NumPy array

    """
    for hospital_address in hospitals_addresses:
        print_weights(hospitals_weights[hospital_address])
    print_weights(averaged_weights)
    
    # computing the similarity factors
    
    factors = dict.fromkeys(hospitals_addresses, 0)
    for hospital_address in hospitals_addresses:
        if SIMILARITY == 'single':
            factors[hospital_address] = similarity_factor_single(
                hospital_address, hospitals_weights, averaged_weights, hospitals_addresses
            )
        else:
            factors[hospital_address] = similarity_factor_multiple(
                hospital_address, hospitals_weights, averaged_weights, hospitals_addresses
            )
    print("SIMILARITY FACTORS: ")
    for hospital_address in factors:
        print(
            f"Hospital address: {hospital_address}\tSimilarity factor: {factors[hospital_address]}"
        )
    
    # computing the AGGREGATION of the collaborators weights
    aggregated_weights = []

    for i in range(weights_dim):
        layer_weights = []
        for hospital_address in hospitals_addresses:
            if SIMILARITY == 'single':
                layer_weights.append(
                    factors[hospital_address] * hospitals_weights[hospital_address][i]
                )
            elif SIMILARITY == 'multiple':
                layer_weights.append(
                    factors[hospital_address][i] * hospitals_weights[hospital_address][i]
                )
        aggregated_weights.append(sum(layer_weights))
    
    for i in range(len(aggregated_weights)):
        aggregated_weights[i] = np.array(
            aggregated_weights[i]
        )  # Convert the list to a NumPy array
    """
    # show the aggregated weights structure
    #print_weights(aggregated_weights)

    # for TEST purposes:    compare AGGREGATED and AVERAGED parameters performance
    aggregated_weights = averaged_weights

    # sending the aggregated parameters on the IPFS and Blockchain
    aggregated_weights_bytes = weights_encoding(aggregated_weights)
    res = IPFS_client.add(aggregated_weights_bytes, pin=PIN_BOOL)

    hash_encoded = res["Hash"].encode("utf-8")
    send_aggregated_weights_tx = FL_contract.send_aggregated_weights(
        hash_encoded, {"from": manager}
    )
    send_aggregated_weights_tx.wait(1)

    test_information(aggregated_weights)

async def starting():
    print("I am the Manager")
    """uploading model and compile information on the Blockchain"""
    encoded_model = get_encoded_model(NUM_CLASSES, "FedProx")
    print("after get_encoded_model")
    transaction_options = {
        "from": manager,
        "gas_limit": 2000000
    }

    send_model_tx = FL_contract.send_model(encoded_model, transaction_options)
    send_model_tx.wait(1)
    print(send_model_tx.events)

    encoded_compile_info = get_encoded_compile_info()

    send_compile_info_tx = FL_contract.send_compile_info(
        encoded_compile_info, {"from": manager}
    )
    send_compile_info_tx.wait(1)
    print(send_compile_info_tx.events)

    # print model details
    model_test.summary()

    # change the contract state to START
    start_tx = FL_contract.start({"from": manager})
    start_tx.wait(1)
    print(start_tx.events)

    # await for the collaborators to retrieve the model
    coroutine_RM = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    print("COROUTINE: waiting 'retrieve_model'\n", coroutine_RM)
    coroutine_result_PM = await coroutine_RM
    assert_coroutine_result(coroutine_result_PM, "retrieve_model")
    print("I waited retrieve_model")
    print_line("_")

    # await for the collaborators to retrieve the compile information
    coroutine_RCI = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    print("COROUTINE: waiting 'retrieve_compile_info'\n", coroutine_RCI)
    coroutine_result_RCI = await coroutine_RCI
    assert_coroutine_result(coroutine_result_RCI, "retrieve_compile_info")
    print("I waited retrieve_compile_info")
    print_line("_")

    # hospitals synchronization
    time.sleep(10)

    # change the contract state to LEARNING
    learning_tx = FL_contract.learning({"from": manager})
    learning_tx.wait(1)
    #print(learning_tx.events)


async def main():
    print("I am the Manager")
    """uploading model and compile information on the Blockchain"""
    encoded_model = get_encoded_model(NUM_CLASSES, model_used)
    print("after get_encoded_model")
    transaction_options = {
        "from": manager,
        "gas_limit": 2000000
    }

    send_model_tx = FL_contract.send_model(encoded_model, transaction_options)
    send_model_tx.wait(1)
    print(send_model_tx.events)

    encoded_compile_info = get_encoded_compile_info()

    send_compile_info_tx = FL_contract.send_compile_info(
        encoded_compile_info, {"from": manager}
    )
    send_compile_info_tx.wait(1)
    print(send_compile_info_tx.events)

    # print model details
    model_test.summary()

    # change the contract state to START
    start_tx = FL_contract.start({"from": manager})
    start_tx.wait(1)
    print(start_tx.events)

    # await for the collaborators to retrieve the model
    coroutine_RM = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    print("COROUTINE: waiting 'retrieve_model'\n", coroutine_RM)
    coroutine_result_PM = await coroutine_RM
    assert_coroutine_result(coroutine_result_PM, "retrieve_model")
    print("I waited retrieve_model")
    print_line("_")

    # await for the collaborators to retrieve the compile information
    coroutine_RCI = contract_events.listen(
        "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_SECONDS
    )
    print("COROUTINE: waiting 'retrieve_compile_info'\n", coroutine_RCI)
    coroutine_result_RCI = await coroutine_RCI
    assert_coroutine_result(coroutine_result_RCI, "retrieve_compile_info")
    print("I waited retrieve_compile_info")
    print_line("_")

    # hospitals synchronization
    time.sleep(10)

    # change the contract state to LEARNING
    learning_tx = FL_contract.learning({"from": manager})
    learning_tx.wait(1)


    for round in range(NUM_ROUNDS):
        print(f"FL ROUND {round+1}...")

        # await for the collaborators to send the weights
        coroutine_SW = contract_events.listen(
            "EveryCollaboratorHasCalledOnlyOnce", timeout=TIMEOUT_DEVICES
        )
        print("COROUTINE: waiting 'send_weights'\n", coroutine_SW)
        coroutine_result_SW = await coroutine_SW
        #assert_coroutine_result(coroutine_result_SW, "send_weights")
        print("Weights arrived")
        print_line("_")

        # reset the weights related events
        reset_weights_tx = FL_contract.reset_weights({"from": manager})
        reset_weights_tx.wait(1)

        # continue after reception
        federated_learning()
        print_line("*")

    # close the BLockchain at the end of the Federated Learning
    close_tx = FL_contract.close({"from": manager})
    close_tx.wait(1)
    print(close_tx.events)

    network.disconnect()

    print("RESULTS - Overall Performance Evaluation through Federated Learning...")
    for round in range(NUM_ROUNDS):
        print(
            f"Round {round+1}:\t Loss: {FL_evaluation[round][0]:.3f} - Accuracy: {FL_evaluation[round][1]:.3f}"
        )
        print(FL_classification_report[round] + "\n")

    sys.exit(0)


asyncio.run(main())
