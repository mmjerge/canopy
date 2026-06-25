"""Real-LLM backing for the routing experiments via Amazon Bedrock.

Turns the synthetic ``PrefixTreeRouting`` quality arrays into *measured* per-model quality
by calling Bedrock models (uniformly, via the Converse API) and grading the responses.
The routing algorithms in :mod:`oco.bandits.routing` are unchanged -- only the ``quality``
matrix comes from real models.

Requires the optional ``llm`` extra (``uv sync --extra llm``) and AWS credentials with
Bedrock invoke permission (e.g. the role provisioned by ``terraform/bedrock.tf``). Model
access must also be enabled in the Bedrock console.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

# Approximate USD price per 1K tokens (input, output); override as needed / per region.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": (0.003, 0.015),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.00025, 0.00125),
    "meta.llama3-1-70b-instruct-v1:0": (0.00099, 0.00099),
    "meta.llama3-1-8b-instruct-v1:0": (0.00022, 0.00022),
    "amazon.nova-pro-v1:0": (0.0008, 0.0032),
    "amazon.nova-lite-v1:0": (0.00006, 0.00024),
    "amazon.nova-micro-v1:0": (0.000035, 0.00014),
    "mistral.mistral-large-2407-v1:0": (0.002, 0.006),
    "mistral.mistral-small-2402-v1:0": (0.001, 0.003),
    "cohere.command-r-plus-v1:0": (0.003, 0.015),
    # cross-region inference-profile IDs
    "us.meta.llama3-1-70b-instruct-v1:0": (0.00072, 0.00072),
    "us.meta.llama3-1-8b-instruct-v1:0": (0.00022, 0.00022),
    "us.meta.llama3-3-70b-instruct-v1:0": (0.00072, 0.00072),
    "us.deepseek.r1-v1:0": (0.00135, 0.0054),
}


class BedrockClient:
    """Thin wrapper over the Bedrock Converse API (uniform across providers).

    Args:
        region: AWS region (must have Bedrock + model access).
        role_arn: Optional role to assume (e.g. the terraform Bedrock app role).
        max_tokens: Default generation cap.
    """

    def __init__(self, region: str = "us-east-1", role_arn: str | None = None,
                 max_tokens: int = 512) -> None:
        import boto3  # lazy: only needed when actually calling Bedrock

        self.max_tokens = max_tokens
        if role_arn:
            sts = boto3.client("sts", region_name=region)
            creds = sts.assume_role(RoleArn=role_arn, RoleSessionName="oco-routing")["Credentials"]
            session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )
        else:
            session = boto3.Session()
        self.runtime = session.client("bedrock-runtime", region_name=region)

    def generate(self, model_id: str, prompt: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> tuple[str, int, int]:
        """Return ``(response_text, input_tokens, output_tokens)`` for one prompt.

        ``temperature`` (if given) enables diverse sampling across calls -- needed for
        best-of-N / rollout search; ``max_tokens`` overrides the client default.
        """
        cfg: dict = {"maxTokens": max_tokens or self.max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        resp = self.runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=cfg,
        )
        text = resp["output"]["message"]["content"][0]["text"]
        usage = resp["usage"]
        return text, int(usage["inputTokens"]), int(usage["outputTokens"])


def measure_quality_matrix(
    prompts: Sequence[str],
    model_ids: Sequence[str],
    client: BedrockClient,
    grade: Callable[[str, str], float],
    pricing: dict[str, tuple[float, float]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build ``(quality, costs)`` for :class:`~oco.bandits.routing.PrefixTreeRouting`.

    For each (model, prompt) it generates a response, grades it to a quality in [0, 1],
    and accumulates token cost. Returns the ``(n_models, n_prompts)`` quality matrix and
    the per-model average cost per query (in USD). ``len(prompts)`` should equal
    ``branching ** depth`` for the routing tree.
    """
    pricing = pricing or DEFAULT_PRICING
    n_m, n_p = len(model_ids), len(prompts)
    quality = np.zeros((n_m, n_p), dtype=np.float64)
    costs = np.zeros(n_m, dtype=np.float64)
    for mi, model_id in enumerate(model_ids):
        in_price, out_price = pricing.get(model_id, (0.001, 0.001))
        total = 0.0
        for pi, prompt in enumerate(prompts):
            text, in_tok, out_tok = client.generate(model_id, prompt)
            quality[mi, pi] = float(np.clip(grade(prompt, text), 0.0, 1.0))
            total += in_price * in_tok / 1000.0 + out_price * out_tok / 1000.0
        costs[mi] = total / max(1, n_p)
    return quality, costs
