"""A minimal Harbor agent that executes a *provided* candidate solution script in the task
container --- the execution primitive for value-guided search on Terminal-Bench.

Terminal-Bench 2.x runs on Harbor (Python >=3.12), whose ``oracle`` agent runs a task's reference
``solve.sh`` and whose verifier then grades the container. We cannot do value-guided *tree* search
by forking a live terminal (filesystem + processes cannot be faithfully cloned --- the ALFWorld
failure mode). Instead we run our method the way it worked on SWE-bench: generate full candidate
solutions and grade each in a FRESH container. This agent is the piece that runs one such
candidate: our orchestrator (``examples/reasoning/terminalbench_search.py``) sets the candidate
script in an environment variable, invokes ``harbor run`` with this agent (one fresh container),
and reads Harbor's verifier results as the cheap probe / leaf grade.

The script is passed as base64 in ``CANOPY_SOLUTION_B64`` to avoid all shell-quoting issues; the
agent decodes it into the container and executes it. Registered with Harbor via its import path:

    harbor run ... -a canopy.bandits.harbor_agent:CanopyScriptAgent

INTEGRATION NOTE: the three marked lines below (upload/exec) are the Harbor API touch points to
verify against the installed harbor version on the box; everything else is plain Python.
"""

from __future__ import annotations

import os
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_SOLUTION_ENV = "CANOPY_SOLUTION_B64"
_CONTAINER_SCRIPT = "/tmp/canopy_solve.sh"


class CanopyScriptAgent(BaseAgent):
    """Runs the base64-encoded solution in ``$CANOPY_SOLUTION_B64`` as a shell script, once."""

    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    @override
    def name() -> str:
        return "canopy-script"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        b64 = os.environ.get(_SOLUTION_ENV, "")
        if not b64:
            self.logger.warning("CANOPY_SOLUTION_B64 not set; running empty solution")
            return
        # Decode the script into the container, then execute it. base64 avoids any quoting issues
        # with arbitrary script content (heredocs, quotes, newlines).
        await environment.exec(  # <-- Harbor API: write the script file in the container
            command=f"echo {b64} | base64 -d > {_CONTAINER_SCRIPT} && chmod +x {_CONTAINER_SCRIPT}",
            user="root",
        )
        await environment.exec(  # <-- Harbor API: run the candidate solution
            command=f"bash {_CONTAINER_SCRIPT}",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
