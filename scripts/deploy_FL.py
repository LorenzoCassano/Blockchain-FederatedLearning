from brownie import FederatedLearning, MockV3Aggregator, network, config, accounts

FORKED_LOCAL_ENVIRONMENTS = ["mainnet-fork", "mainnet-fork-dev"]
LOCAL_BLOCKCHAIN_ENVIRONMENTS = ["development", "fl-local"]

DECIMALS = 8
STARTING_PRICE = 200000000000  # this is 2,000


def get_account():
    #return accounts.add(config["wallets"]["from_key"])
    if (
        network.show_active() in LOCAL_BLOCKCHAIN_ENVIRONMENTS
        or network.show_active() in FORKED_LOCAL_ENVIRONMENTS
    ):
        return accounts[0]
    else:
        return accounts.add(config["wallets"]["from_key"])



def deploy_mocks():
    """
    Use this script if you want to deploy mocks to a testnet
    """
    print(f"The active network is {network.show_active()}")
    print("Deploying Mocks...",len(MockV3Aggregator))
    gas_used = 0
    if len(MockV3Aggregator) <= 0:
        trans = MockV3Aggregator.deploy(DECIMALS, STARTING_PRICE, {"from": get_account()})
        gas_used = trans.tx.gas_used
    print("Mocks Deployed!")
    return gas_used


def deploy_federated_learning(gas_cons_setup):
    account = get_account()
    gas_used = 0
    if network.show_active() not in LOCAL_BLOCKCHAIN_ENVIRONMENTS:
        price_feed_address = config["networks"][network.show_active()][
            "eth_usd_price_feed"
        ]
    else:
        gas_used = deploy_mocks()
        price_feed_address = MockV3Aggregator[-1].address

    federated_learning = FederatedLearning.deploy(
        {"from": account},
        publish_source=config["networks"][network.show_active()].get("verify"),
    )
    print(f"Contract deployed to {federated_learning.address}")
    gas_cons_setup['deploy_mocks_fee'] = gas_used + federated_learning.tx.gas_used
    return federated_learning


def main():
    deploy_federated_learning()
