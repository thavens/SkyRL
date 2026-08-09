"""Unit tests for proto response serialization (skyrl/tinker/proto_conv.py)."""

import numpy as np
import pytest

from skyrl.tinker import proto_conv, types

SAMPLE_RESULT = {
    "sequences": [
        {"stop_reason": "stop", "tokens": [1, 2, 3], "logprobs": [-0.1, -0.2, -0.3]},
        {"stop_reason": "length", "tokens": [4, 5], "logprobs": [-0.4, -0.5]},
    ],
    "prompt_logprobs": None,
    "topk_prompt_logprobs": None,
}


def test_external_uses_sample_serializer():
    """Samples forwarded to an external engine are stored under EXTERNAL but hold
    the same SampleOutput payload; SDK >=0.24 rejects JSON for SampleResponse, so
    EXTERNAL must serialize exactly like SAMPLE (this regressed once: the local
    vLLM-forwarding deployment serves *only* EXTERNAL samples)."""
    assert proto_conv._SERIALIZERS[types.RequestType.EXTERNAL] is proto_conv._SERIALIZERS[types.RequestType.SAMPLE]


def test_serialize_sample_and_external_round_trip():
    pb = proto_conv._proto_module()
    if pb is None:
        pytest.skip("installed tinker SDK has no proto module")
    sample_bytes = proto_conv.serialize(types.RequestType.SAMPLE, SAMPLE_RESULT)
    external_bytes = proto_conv.serialize(types.RequestType.EXTERNAL, SAMPLE_RESULT)
    assert sample_bytes is not None and sample_bytes == external_bytes

    decoded = pb.SampleResponse()
    decoded.ParseFromString(external_bytes)
    assert len(decoded.sequences) == 2
    # Tokens ride as raw little-endian int32 bytes (see _sample_response).
    assert np.frombuffer(decoded.sequences[0].tokens, dtype=np.int32).tolist() == [1, 2, 3]
    assert decoded.sequences[1].stop_reason == pb.STOP_REASON_LENGTH


def test_unsupported_type_returns_none():
    assert proto_conv.serialize(types.RequestType.OPTIM_STEP, {"anything": 1}) is None


def test_deserialize_forward_backward_request_round_trip():
    """SDK >=0.25 posts proto bodies to /api/v1/forward_backward. Decode one
    (built with the proto schema the SDK itself installs) back into the JSON
    request shape and check it validates through the API's pydantic model."""
    pb = proto_conv._proto_module()
    if pb is None:
        pytest.skip("installed tinker SDK has no proto module")

    msg = pb.ForwardBackwardRequest(model_id="model_test", seq_id=1, loss_fn="cross_entropy", forward_only=True)
    datum = msg.data.add()
    chunk = datum.model_input.add()
    chunk.encoded_text.tokens = np.asarray([5, 6, 7], dtype=np.int32).tobytes()
    datum.loss_fn_inputs["target_tokens"].dtype = pb.DTYPE_INT64
    datum.loss_fn_inputs["target_tokens"].dense = np.asarray([6, 7, 8], dtype=np.int64).tobytes()
    datum.loss_fn_inputs["weights"].dtype = pb.DTYPE_FLOAT32
    datum.loss_fn_inputs["weights"].dense = np.asarray([0.0, 1.0, 1.0], dtype=np.float32).tobytes()

    decoded = proto_conv.deserialize_forward_backward_request(msg.SerializeToString())
    assert decoded is not None
    request_dict, forward_only = decoded
    assert forward_only is True
    assert request_dict["model_id"] == "model_test"

    from skyrl.tinker.api import ForwardBackwardRequest

    request = ForwardBackwardRequest.model_validate(request_dict)
    assert request.forward_backward_input.loss_fn == "cross_entropy"
    [parsed] = request.forward_backward_input.data
    assert parsed.model_input.chunks[0].tokens == [5, 6, 7]
    assert parsed.loss_fn_inputs["target_tokens"].data == [6, 7, 8]
    assert parsed.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0]
