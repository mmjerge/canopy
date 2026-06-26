"""Amazon Bedrock implementation of :class:`~canopy.llm.base.LLMClient`.

Calls models uniformly via the Bedrock Converse API and reports token-accurate usage, so the
routing / prompt-trimming experiments can swap synthetic quality arrays for measured ones
without changing any algorithm.

Requires the optional ``llm`` extra (``uv sync --extra llm``) and AWS credentials with Bedrock
invoke permission (e.g. the role provisioned by ``terraform/bedrock.tf``). Model access must
also be enabled in the Bedrock console.
"""

from __future__ import annotations

from typing import Any

from canopy.llm.base import Generation

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
        pricing: Optional ``model_id -> (in, out)`` per-1K-token price table.
        runtime: Optional pre-built ``bedrock-runtime`` client (for injection/testing).
    """

    def __init__(
        self,
        region: str = "us-east-1",
        role_arn: str | None = None,
        max_tokens: int = 512,
        pricing: dict[str, tuple[float, float]] | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.pricing = pricing or DEFAULT_PRICING
        if runtime is not None:
            self.runtime = runtime
            return
        import boto3  # lazy: only needed when actually calling Bedrock

        if role_arn:
            sts = boto3.client("sts", region_name=region)
            creds = sts.assume_role(RoleArn=role_arn, RoleSessionName="canopy-routing")[
                "Credentials"
            ]
            session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )
        else:
            session = boto3.Session()
        self.runtime = session.client("bedrock-runtime", region_name=region)

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generation:
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

    def price_per_1k(self, model_id: str) -> tuple[float, float]:
        """Per-1K-token ``(input, output)`` USD price for ``model_id`` (0 if unknown)."""
        return self.pricing.get(model_id, (0.001, 0.001))
