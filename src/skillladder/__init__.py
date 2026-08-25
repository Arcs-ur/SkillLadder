"""Public SkillLadder package surface."""

from fedskill import __version__
from fedskill.models.skill import Skill
from fedskill.server.aggregator import FedSkillServer

SkillLadderServer = FedSkillServer

__all__ = ["Skill", "SkillLadderServer", "__version__"]
