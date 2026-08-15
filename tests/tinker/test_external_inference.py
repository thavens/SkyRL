"""Unit tests for the payload ExternalInferenceClient sends to vLLM."""

import asyncio

import pytest

from skyrl.tinker import api, types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra.external_inference import ExternalInferenceClient


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        n = self._payload["n"]
        return {
            "choices": [
                {
                    "token_ids": [1, 2, 3],
                    "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3]},
                    "finish_reason": "stop",
                }
                for _ in range(n)
            ]
        }


class _CapturingClient:
    """Stands in for the httpx client, recording the JSON body of each POST."""

    def __init__(self):
        self.payloads: list[dict] = []

    async def post(self, url, json=None, headers=None):  # noqa: A002 - mirrors httpx's signature
        self.payloads.append(json)
        return _FakeResponse(json)


class _Prompt:
    """api.ModelInput exposes .to_types(); only that is needed here."""

    @staticmethod
    def to_types() -> types.ModelInput:
        return types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[10, 11, 12])])


class _Request:
    """The subset of api.SampleRequest that _forward_to_engine reads.

    Note the sampling params are the *API-layer* model: on the external path the raw
    `stop` list reaches the forwarder without api.SamplingParams.to_types() having run.
    """

    def __init__(self, sampling_params: api.SamplingParams):
        self.prompt = _Prompt()
        self.num_samples = 1
        self.sampling_params = sampling_params
        self.prompt_logprobs = False
        self.topk_prompt_logprobs = 0
        self.sampling_session_id = None
        self.seq_id = None


def _forward(sampling_params: api.SamplingParams) -> dict:
    """Run one base-model forward and return the payload that reached the engine."""
    config = EngineConfig(base_model="base-model", external_inference_url="http://engine")
    client = ExternalInferenceClient(config, db_engine=None)
    http_client = _CapturingClient()
    asyncio.run(
        client._forward_to_engine(
            _Request(sampling_params),
            model_id="m",
            checkpoint_id="c",
            http_client=http_client,
            base_model="base-model",
        )
    )
    return http_client.payloads[0]


def _params(stop=None) -> api.SamplingParams:
    return api.SamplingParams(temperature=1.0, max_tokens=8, seed=0, stop=stop)


class TestStopConditionsAreForwarded:
    """Stop conditions must reach the engine.

    Dropping them fails silently rather than loudly: every completion just runs to
    max_tokens, and an RL loop then trains at full weight on whatever the model emitted
    past the intended stop. The in-process generator honours both stop_strings and
    stop_tokens, so omitting them here also makes the two sampling paths disagree.
    """

    def test_stop_strings_are_forwarded(self):
        payload = _forward(_params(stop=["<|im_end|>", "\n\nUser:"]))
        assert payload["stop"] == ["<|im_end|>", "\n\nUser:"]

    def test_stop_tokens_are_forwarded(self):
        payload = _forward(_params(stop=[151645, 151643]))
        assert payload["stop_token_ids"] == [151645, 151643]

    def test_stop_token_is_kept_in_the_output(self):
        """Matches find_string_stop_position, which truncates *after* the stop token."""
        payload = _forward(_params(stop=[248046]))
        assert payload["include_stop_str_in_output"] is True

    @pytest.mark.parametrize("field", ["stop", "stop_token_ids", "include_stop_str_in_output"])
    def test_absent_when_not_requested(self, field):
        """No stop conditions set must not silently impose one."""
        assert field not in _forward(_params())
