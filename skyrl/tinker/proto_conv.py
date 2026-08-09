"""Proto serialization of sample / forward-backward results.

From tinker SDK 0.24 the client decodes ``SampleResponse`` and
``ForwardBackwardOutput`` from protobuf *only* -- a JSON body for either type
raises "the server predates proto response serialization for this type". This
module encodes the JSON result rows this server already stores into the wire
format ``tinker.proto.response_conv`` expects, so both SDK generations work.

The proto schema comes from the installed ``tinker`` package rather than a
vendored copy, so it cannot drift from the client's decoder. It is imported
lazily: nothing else in the server requires the SDK, and without it the server
simply keeps answering in JSON.
"""

from __future__ import annotations

import functools

import numpy as np

from skyrl.tinker import types

# Empty top-k cells, per the SDK's dense-matrix contract.
_MASK_TOKEN_ID = 0
_MASK_LOGPROB = -99999.0

# The public wire carries only these two; the SDK collapses int32/bfloat16 into
# them on the way back out. Mirrors ``tinker.proto.request_conv``'s private maps.
_NUMPY_DTYPES = {"float32": np.float32, "int64": np.int64}


@functools.cache
def _proto_module():
    """The installed SDK's generated proto module, or None when the SDK is absent."""
    try:
        from tinker.proto import tinker_public_pb2

        return tinker_public_pb2
    except ImportError:
        return None


def _sample_response(pb, result: dict) -> bytes:
    stop_reasons = {"stop": pb.STOP_REASON_STOP, "length": pb.STOP_REASON_LENGTH}

    sequences = []
    for seq in result.get("sequences") or []:
        stop_reason = stop_reasons.get(seq.get("stop_reason"))
        if stop_reason is None:
            raise ValueError(f"Unknown stop_reason {seq.get('stop_reason')!r}")
        logprobs = seq.get("logprobs")
        sequences.append(
            pb.SampledSequence(
                stop_reason=stop_reason,
                tokens=np.asarray(seq.get("tokens") or [], dtype=np.int32).tobytes(),
                logprobs=(np.asarray(logprobs, dtype=np.float32).tobytes() if logprobs else None),
            )
        )

    message = pb.SampleResponse(sequences=sequences)

    # Present only in newer SDK schemas; set it from whatever is installed rather than
    # assuming, so one encoder serves both SDK generations.
    if "prompt_cache_hit_tokens" in pb.SampleResponse.DESCRIPTOR.fields_by_name:
        message.prompt_cache_hit_tokens = int(result.get("prompt_cache_hit_tokens") or 0)

    # A scalar per prompt position; None means "not computed" and rides as NaN
    # (numpy casts None to NaN on the way into a float array).
    prompt_logprobs = result.get("prompt_logprobs")
    if prompt_logprobs:
        message.prompt_logprobs = np.asarray(prompt_logprobs, dtype=np.float32).tobytes()

    topk = result.get("topk_prompt_logprobs")
    k = max((len(row) for row in topk or [] if row), default=0)
    if k:
        # Rows can be ragged (vLLM appends the actual token when it misses the
        # top-k) so densify to the widest row; short rows keep the sentinel fill.
        n = len(topk)
        token_ids = np.full((n, k), _MASK_TOKEN_ID, dtype=np.int32)
        logprobs = np.full((n, k), _MASK_LOGPROB, dtype=np.float32)
        for i, row in enumerate(topk):
            if row:
                token_ids[i, : len(row)], logprobs[i, : len(row)] = zip(*row)
        message.topk_prompt_logprobs.CopyFrom(
            pb.TopkPromptLogprobs(
                token_ids=token_ids.tobytes(),
                logprobs=logprobs.tobytes(),
                k=k,
                prompt_length=n,
            )
        )

    return message.SerializeToString()


def _batched_tensor(pb, per_datum: list[dict]):
    """One field's per-datum tensors concatenated, with byte offsets. The SDK
    divides offsets by the dtype's itemsize, so they are byte- not element-
    indexed."""
    dtype = per_datum[0].get("dtype", "float32")
    if dtype not in _NUMPY_DTYPES:
        raise ValueError(f"Unsupported tensor dtype {dtype!r}")
    np_dtype = _NUMPY_DTYPES[dtype]
    proto_dtype = pb.DTYPE_FLOAT32 if dtype == "float32" else pb.DTYPE_INT64

    arrays = [np.asarray(t.get("data") or [], dtype=np_dtype) for t in per_datum]
    offsets = np.cumsum([0] + [a.nbytes for a in arrays], dtype=np.int64)

    # shape is [leading, *trailing]; the SDK recovers leading from the slice size.
    shape = per_datum[0].get("shape") or []
    return pb.BatchedTensor(
        data=b"".join(a.tobytes() for a in arrays),
        offsets=offsets.tobytes(),
        dtype=proto_dtype,
        trailing_shape=list(shape[1:]),
    )


def _forward_backward_output(pb, result: dict) -> bytes:
    outputs = result.get("loss_fn_outputs") or []
    metrics = {k: float(v) for k, v in (result.get("metrics") or {}).items()}
    type_tag = result.get("loss_fn_output_type") or "ArrayRecord"

    records = []
    if outputs:
        # Field names are invariant across datums, so one ArrayRecord carries
        # the whole batch -- the shape the SDK's v1 writer emits.
        fields = {name: _batched_tensor(pb, [datum[name] for datum in outputs]) for name in outputs[0]}
        records.append(pb.ArrayRecord(type_tag=type_tag, fields=fields, num_datums=len(outputs)))

    return pb.ForwardBackwardOutput(
        loss_fn_output_type=type_tag,
        loss_fn_outputs=records,
        metrics=metrics,
    ).SerializeToString()


# Only these request types have a proto encoding; everything else stays JSON.
# Upstream source of truth is ``response_conv.PROTO_SUPPORTED_TYPES`` -- if the
# SDK adds a type there, it has to be added here too.
_SERIALIZERS = {
    types.RequestType.SAMPLE: _sample_response,
    # Samples forwarded to an external engine (vLLM) are stored under EXTERNAL
    # but carry the same SampleOutput payload; SDK >=0.24 requires proto for
    # them just the same.
    types.RequestType.EXTERNAL: _sample_response,
    types.RequestType.FORWARD_BACKWARD: _forward_backward_output,
    types.RequestType.FORWARD: _forward_backward_output,
}


def _tensor_to_list(pb, tensor) -> list:
    """Decode a proto Tensor into the flat list the JSON TensorData model takes.

    The public wire carries float32/int64 (request_conv collapses everything
    else before sending). Only dense encoding is supported here: the sparse-CSR
    path exists on the wire for topk-logprob matrices, which never appear in
    forward_backward loss_fn_inputs.
    """
    if tensor.WhichOneof("encoding") == "sparse_csr":
        raise ValueError("sparse_csr loss_fn_inputs are not supported by this server")
    np_dtype = {pb.DTYPE_FLOAT32: np.float32, pb.DTYPE_INT64: np.int64}.get(tensor.dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported tensor dtype on the wire: {tensor.dtype}")
    return np.frombuffer(tensor.dense, dtype=np_dtype).tolist()


def deserialize_forward_backward_request(body: bytes) -> tuple[dict, bool] | None:
    """Decode a proto ``ForwardBackwardRequest`` body (SDK >=0.25 sends these to
    /api/v1/forward_backward with Content-Type application/x-protobuf) into the
    JSON-equivalent request dict, plus its ``forward_only`` flag.

    Returns None when the installed SDK has no proto module, in which case the
    caller cannot accept proto bodies at all.
    """
    pb = _proto_module()
    if pb is None:
        return None
    msg = pb.ForwardBackwardRequest()
    msg.ParseFromString(body)

    data = []
    for datum in msg.data:
        chunks = []
        for chunk in datum.model_input:
            if chunk.WhichOneof("chunk") != "encoded_text":
                raise ValueError("Only encoded_text model input chunks are supported (text-only server)")
            chunks.append(
                {
                    "type": "encoded_text",
                    "tokens": np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist(),
                }
            )
        data.append(
            {
                "model_input": {"chunks": chunks},
                "loss_fn_inputs": {name: {"data": _tensor_to_list(pb, t)} for name, t in datum.loss_fn_inputs.items()},
            }
        )

    request = {
        "model_id": msg.model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": msg.loss_fn,
            "loss_fn_config": dict(msg.loss_fn_config) or None,
        },
    }
    return request, bool(msg.forward_only)


def serialize(request_type: types.RequestType, result: dict) -> bytes | None:
    """Proto bytes for a stored result row, or None if the type has no proto
    encoding (or the SDK is not installed) and it should be returned as JSON."""
    serializer = _SERIALIZERS.get(request_type)
    if serializer is None:
        return None
    pb = _proto_module()
    if pb is None:
        return None
    return serializer(pb, result)
