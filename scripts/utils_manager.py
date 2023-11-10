import numpy as np
import json
import io
from fedAvg import FedAvg
from fedProx import FedProx

WIDTH = 176
HEIGHT = 208
DEPTH = 1
NUM_CLASSES = 4

NUM_ROUNDS = 3

TIMEOUT_SECONDS = 600
EPSILON = 10 ** (-5)

# similarity = ['single', 'multiple', 'averaged']
SIMILARITY = "single"

compile_info = {
    "optimizer": "Adam",
    "metrics": ["accuracy"],
}


def get_encoded_compile_info():
    JSON_compile_info = json.dumps(compile_info)
    encoded_compile_info = JSON_compile_info.encode("utf-8")
    return encoded_compile_info


def get_encoded_model(num_classes, approach = "FedAvg"):
    if approach == "FedAvg":
        model = FedAvg(num_classes)
    else:
        model = FedProx(num_classes)

    JSON_model = model.to_json()
    encoded_model = JSON_model.encode("utf-8")
    return encoded_model


def assert_coroutine_result(_coroutine_result, _function_name):
    if _coroutine_result.event_data.args.functionName == _function_name:
        print(f'The event "{_function_name}" has been correctly catched')
    else:
        raise Exception('ERROR: event "', _function_name, '" not catched')


def weights_encoding(_weights):
    weights_listed = [param.tolist() for param in _weights]
    weights_JSON = json.dumps(weights_listed)
    weights_encoded = weights_JSON.encode("utf-8")
    weights_bytes = io.BytesIO(weights_encoded)
    return weights_bytes


def weights_decoding(_weights_encoded):
    weights_JSON = _weights_encoded.decode("utf-8")
    weights_listed = json.loads(weights_JSON)
    weights = [np.array(param, dtype=np.float32) for param in weights_listed]
    return weights


def similarity_single(
    _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
):
    numerator = [
        np.linalg.norm(h_w - a_w)
        for hospital_address in _hospitals_addresses
        for h_w, a_w in zip(_hospitals_weights[hospital_address], _averaged_weights)
    ]
    numerator = sum(numerator)

    denominator = [
        np.linalg.norm(h_w - a_w)
        for h_w, a_w in zip(_hospitals_weights[_hospital_address], _averaged_weights)
    ]
    denominator = sum(denominator) + (10**-5)

    result = numerator / denominator
    return result



def similarity_factor_single(
    _hospital_address, _hospital_weights, _averaged_weights, _hospitals_addresses
):
    return similarity_single(
        _hospital_address, _hospital_weights, _averaged_weights, _hospitals_addresses
    ) / sum(
        [
            similarity_single(
                hospital_address,
                _hospital_weights,
                _averaged_weights,
                _hospitals_addresses,
            )
            for hospital_address in _hospitals_addresses
        ]
    )



# utility function to compute the Frobenius norm between 2 matrices
def frobenius_norm(_hospital_address, _hospitals_weights, _averaged_weights):
    result = [
        np.linalg.norm(h_w - a_w)
        for h_w, a_w in zip(_hospitals_weights[_hospital_address], _averaged_weights)
    ]
    return np.array(result)


# utility function to compute the similarity factor used in the aggregated weights
def similarity_multiple(
    _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
):
    distances = [
        frobenius_norm(hospital_address, _hospitals_weights, _averaged_weights)
        for hospital_address in _hospitals_addresses
    ]

    numerator = [sum(layer) for layer in zip(*distances)]
    denominator = (
        frobenius_norm(_hospital_address, _hospitals_weights, _averaged_weights)
        + EPSILON
    )

    result = np.divide(numerator, denominator)
    return result


# return the weighted contribute of a single collaborator on a FL round
def similarity_factor_multiple(
    _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
):
    similarities = [
        similarity_multiple(
            hospital_address,
            _hospitals_weights,
            _averaged_weights,
            _hospitals_addresses,
        )
        for hospital_address in _hospitals_addresses
    ]

    numerator = similarity_multiple(
        _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
    )

    denominator = [sum(layer) for layer in zip(*similarities)]

    result = np.divide(numerator, denominator)
    return result




