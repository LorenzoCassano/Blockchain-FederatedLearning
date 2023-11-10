import numpy as np
import json
import io

NUM_EPOCHS = 5
BATCH_SIZE = 64


def decode_utf8(_tx):
    retrieved_utf8 = _tx.return_value
    return retrieved_utf8.decode("utf-8")


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
