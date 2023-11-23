import os
import sys

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils_simulation import createHospitals, set_hospitals, get_hospitals, print_line, generate_random_split,print_hospital_split

from brownie import FederatedLearning, accounts
import deploy_FL
from constants import *
import json

print("Private Key: ",os.getenv(("PRIVATE_KEY")))
# with this CLI argument choose to retrieve or to create the hospitals information

# CMD ARGUMENTS
# default values
isCreated = True
if "main" in sys.argv:
    isCreated = False


def main(dataset="",number_device=3):
    """
    1)  Hospitals creation
    """
    if "brain_tumor" in sys.argv:
        train_path = DATASET_TRAIN_PATH_TUM
        test_path = DATASET_TEST_PATH_TUM
    else:
        train_path = DATASET_TRAIN_PATH_ALZ
        test_path = DATASET_TEST_PATH_ALZ
    hospitals = None

    if isCreated:
        print("Loading dataset from pkl files")
        hospitals = get_hospitals()
    else:
        n = int(number_device)
        hospital_split = generate_random_split(n)
        print(f"Creating dataset from {train_path} with {n} devices")
        print_hospital_split(hospital_split)
        # saving the file

        with open(HOSPITAL_SPLIT_FILE, 'w') as json_file:
            json.dump(hospital_split, json_file)
        hospitals = createHospitals(train_path,test_path,hospital_split)
    print("[1]\tHospitals have been created")
    print_line("*")

    """
    2)  KYC Process and Off-Chain Hospitals Registration
        - This must be done before the blockchain
    """
    print("[2]\tKYC Process and Off-Chain Hospitals Registration completed")
    print_line("*")

    """
    3)  Blockchain implementation
    """
    deploy_FL.deploy_federated_learning()
    print("[3]\tFederatedLearning contract has been deployed - Blockchain implemented")
    print_line("*")

    """
    4)  Assign Blockchain addresses to Hospitals
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
    print("[4]\tAssigned Ganache addresses to the hospitals")
    print_line("*")

    """
    5)  Opening the Blockchain and adding the Collaborators
    """
    federated_learning = FederatedLearning[-1]
    manager = deploy_FL.get_account()
    print("Manager address:", manager)

    open_tx = federated_learning.open({"from": manager})
    open_tx.wait(1)
    print(open_tx.events)

    for hospital_name in hospitals:
        hospital_address = hospitals[hospital_name].address
        add_tx = federated_learning.add_collaborator(
            hospital_address, {"from": manager}
        )
        add_tx.wait(1)
    print("[5]\tBlockchain opened and collaborators authorized to use it")
    print_line("*")

    for key, hospital in hospitals.items():
        print(f"Key: {key}")
        print()
        # set the Hospital file
        set_hospitals(hospitals)


if __name__ == "__main__":
    main()
