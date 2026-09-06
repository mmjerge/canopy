"""Amazon Bedrock implementation of :class:`~canopy.llm.base.LLMClient`.

Calls models uniformly via the Bedrock Converse API and reports token-accurate usage, so the
routing / prompt-trimming experiments can swap synthetic quality arrays for measured ones
without changing any algorithm.

Requires the optional ``llm`` extra (``uv sync --extra llm``) and AWS credentials with Bedrock
invoke permission (e.g. the role provisioned by ``terraform/bedrock.tf``). Model access must
also be enabled in the Bedrock console.
"""

from __future__ import annotations

import random
import time
from typing import Any

from canopy.llm.base import Generation

# Transient Bedrock errors worth retrying with backoff (vs. a hard config/access error). Matched
# by exception class name so we need not import each botocore error factory type.
_TRANSIENT_ERRORS = frozenset(
    {
        "ThrottlingException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
        "ServiceQuotaExceededException",
        "TooManyRequestsException",
    }
)

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
        max_retries: int = 6,
    ) -> None:
        self.max_tokens = max_tokens
        self.pricing = pricing or DEFAULT_PRICING
        self.region = region
        self.max_retries = max_retries
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
        seed: int | None = None,
    ) -> Generation:
        """Return ``(response_text, input_tokens, output_tokens)`` for one prompt.

        ``temperature`` (if given) enables diverse sampling across calls -- needed for
        best-of-N / rollout search; ``max_tokens`` overrides the client default. ``seed``
        distinguishes independent same-prompt samples for caching; the Converse API samples
        stochastically at ``temperature`` (it has no portable seed field), so it is used only to
        keep draws distinct in any wrapping cache, not forwarded to the model.
        """
        cfg: dict = {"maxTokens": max_tokens or self.max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        # Bedrock throttles and occasionally times out under load; retry transient errors with
        # exponential backoff + jitter so one flaky call doesn't abort a long experiment run.
        resp = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.runtime.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig=cfg,
                )
                break
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ not in _TRANSIENT_ERRORS or attempt == self.max_retries:
                    raise
                time.sleep(min(2.0**attempt + random.random(), 30.0))
        # The loop above always either breaks with a response or raises on the final retry.
        assert resp is not None
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        # Concatenate all text blocks; reasoning models (e.g. DeepSeek R1) also emit
        # non-text "reasoningContent" blocks, which carry no "text" key and are skipped.
        text = "".join(b["text"] for b in blocks if isinstance(b, dict) and "text" in b)
        usage = resp["usage"]
        return text, int(usage["inputTokens"]), int(usage["outputTokens"])

    def price_per_1k(self, model_id: str) -> tuple[float, float]:
        """Per-1K-token ``(input, output)`` USD price for ``model_id`` (0 if unknown)."""
        return self.pricing.get(model_id, (0.001, 0.001))

    def list_text_models(self, only_available: bool = True) -> list[str]:
        """Discover invokable text model / inference-profile IDs in this region.

        Returns cross-region inference-profile IDs (directly usable as ``modelId`` in the
        Converse API) plus on-demand foundation text models. When ``only_available`` is set
        (the default), each candidate is checked with ``GetFoundationModelAvailability`` and
        kept only if it is authorized, entitled, and available in this region; this avoids
        attempting catalog IDs that would raise ``ResourceNotFoundException`` (not in region /
        not entitled) or ``AccessDeniedException`` (not authorized). Requires the ``bedrock``
        (control-plane) permissions; if a check cannot be performed, the candidate is kept and
        left for the runtime call to resolve.
        """
        import boto3

        ctrl = boto3.client("bedrock", region_name=self.region)
        # (invoke_id, base_model_id) -- base id is what availability is checked against.
        candidates: list[tuple[str, str]] = []
        try:
            resp = ctrl.list_inference_profiles(maxResults=100)
            for prof in resp.get("inferenceProfileSummaries", []):
                if prof.get("status", "ACTIVE") != "ACTIVE":
                    continue
                pid = prof["inferenceProfileId"]
                models = prof.get("models") or []
                base = pid
                if models:
                    arn = models[0].get("modelArn", "")
                    base = arn.split("/")[-1] or pid
                candidates.append((pid, base))
        except Exception:  # noqa: BLE001 -- control-plane perms may be missing; fall back
            pass
        try:
            fm = ctrl.list_foundation_models(byOutputModality="TEXT")
            for m in fm.get("modelSummaries", []):
                if m.get("modelLifecycle", {}).get("status") != "ACTIVE":
                    continue
                if "ON_DEMAND" not in m.get("inferenceTypesSupported", []):
                    continue
                if "TEXT" not in m.get("inputModalities", []):
                    continue
                candidates.append((m["modelId"], m["modelId"]))
        except Exception:  # noqa: BLE001
            pass

        seen: set[str] = set()
        uniq: list[tuple[str, str]] = []
        for inv, base in candidates:
            if inv not in seen:
                seen.add(inv)
                uniq.append((inv, base))
        if not only_available:
            return [inv for inv, _ in uniq]

        out: list[str] = []
        for inv, base in uniq:
            av = self.model_availability(base)
            if av is None:
                out.append(inv)  # could not determine; let the runtime call decide
            elif (
                av["authorizationStatus"] == "AUTHORIZED"
                and av["regionAvailability"] == "AVAILABLE"
                and av["entitlementAvailability"] == "AVAILABLE"
            ):
                out.append(inv)
        return out

    def model_availability(self, model_id: str) -> dict[str, str] | None:
        """Return authorization / entitlement / region status for ``model_id``.

        Returns a dict with ``authorizationStatus``, ``entitlementAvailability``, and
        ``regionAvailability`` (each an enum string), or ``None`` if the status cannot be
        determined (e.g. missing ``bedrock:GetFoundationModelAvailability`` permission, or the
        ID is an inference profile that the API does not resolve).
        """
        import boto3

        ctrl = boto3.client("bedrock", region_name=self.region)
        try:
            r = ctrl.get_foundation_model_availability(modelId=model_id)
        except Exception:  # noqa: BLE001
            return None
        return {
            "authorizationStatus": r.get("authorizationStatus", ""),
            "entitlementAvailability": r.get("entitlementAvailability", ""),
            "regionAvailability": r.get("regionAvailability", ""),
        }
