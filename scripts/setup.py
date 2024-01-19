import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils_simulation import createHospitals, set_hospitals, get_hospitals, print_line, generate_random_split,print_hospital_split, save_dataset
from classHospital import Hospital
from brownie import FederatedLearning, accounts
import deploy_FL
from constants import *
import json

isCreated = True
if "main" in sys.argv:
    isCreated = False



def main(*args, **kwargs):
    if "brain_tumor" in sys.argv:
        train_path = DATASET_TRAIN_PATH_TUM
        test_path = DATASET_TEST_PATH_TUM
        dataset_name = BRAIN_TUMOR
    else:
        train_path = DATASET_TRAIN_PATH_ALZ
        test_path = DATASET_TEST_PATH_ALZ
        dataset_name = ALZHEIMER
    hospitals = None


    """
    1)  KYC Process and Off-Chain Hospitals Registration
        - This must be done before the blockchain
    """
    if isCreated:
        print("Loading dataset from pkl files")
        hospitals = get_hospitals()
    else:
        hospital_split = generate_random_split(NUM_DEVICES)
        print(f"Creating dataset from {train_path} with {NUM_DEVICES} devices")
        print()
        print_hospital_split(hospital_split)
        # saving the file

        with open(HOSPITAL_SPLIT_FILE, 'w') as json_file:
            json.dump(hospital_split, json_file)
        hospitals, hospital_dataset = createHospitals(train_path,test_path,hospital_split,dataset_name)
        print("Saving dataset to pkl files...")
        print()
        save_dataset(hospital_dataset)
        print("Dataset saved to pkl files!")

    gas_cons_setup = {}

    print("[1]\tKYC Process and Off-Chain Hospitals Registration completed")
    print_line("*")
    print('\n'*2)


    """
    2)  Blockchain implementation
    """
    deploy_FL.deploy_federated_learning(gas_cons_setup)
    print("[2]\tFederatedLearning contract has been deployed - Blockchain implemented")
    print_line("*")
    print('\n' * 2)

    """
    3)  Assign Blockchain addresses to Hospitals
    """
    # only with Ganache fl-local network
    for idx, hospital_name in enumerate(hospitals, start=1):
        hospitals[hospital_name].address = accounts[idx].address
        print(
            "Hospital name:",
            hospital_name,
            "\tGanache address:",
            hospitals[hospital_name].address,
            "\tGanache idx:",
            idx,
        )
    print("[3]\tAssigned Ganache addresses to the hospitals")
    print_line("*")
    print('\n' * 2)

    """
    4)  Opening the Blockchain 
    """
    federated_learning = FederatedLearning[-1]
    manager = deploy_FL.get_account()
    print("Manager address:", manager)

    open_tx = federated_learning.open({"from": manager})
    gas_cons_setup['open_blockchain_fee'] = open_tx.gas_used
    open_tx.wait(1)
    print("[4]\tBlockchain opened ")
    print_line("*")
    print('\n' * 2)

    """
    5) Adding collaborator
    """
    gas_cons_setup['add_collaborator_fee'] = 0
    for hospital_name in hospitals:
        hospital_address = hospitals[hospital_name].address
        add_tx = federated_learning.add_collaborator(
            hospital_address, {"from": manager}
        )
        gas_cons_setup['add_collaborator_fee'] += add_tx.gas_used
        add_tx.wait(1)

    set_hospitals(hospitals)
    print("[5]\tBlockchain opened and collaborators authorized to use it")
    print_line("*")
    print('\n' * 2)

    """
    6)  saving the gas consumption setup
    """
    with open("gas_fee_setup.json", 'w') as json_file:
        json.dump(gas_cons_setup, json_file)

    print("[6]\tGas consumption setup saved")
    print_line("*")


if __name__ == "__main__":
    main()
