"""Task client — represents a single user / task node in the federation."""

from __future__ import annotations

from dataclasses import dataclass, field
from fedskill.models.skill import Skill
from fedskill.extractor.skill_extractor import extract_skills


@dataclass
class TaskClient:
    """A federated client that holds task data and extracts local skills.

    Each client corresponds to one task (A / B / C / D …) with its own
    traces and resulting local skill set.
    """

    task_name: str
    task_description: str
    traces: list[str] = field(default_factory=list)
    local_skills: list[Skill] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def add_trace(self, trace: str) -> None:
        """Append an execution trace."""
        self.traces.append(trace)

    def extract_local_skills(self) -> list[Skill]:
        """Run skill extraction on local traces and store the results."""
        self.local_skills = extract_skills(
            task_name=self.task_name,
            task_description=self.task_description,
            traces=self.traces,
        )
        return self.local_skills

    def upload_skills(self) -> list[Skill]:
        """Return local skills ready to be sent to the aggregation server."""
        return list(self.local_skills)
