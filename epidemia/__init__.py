__all__ = [
    '__author__', '__description__', '__email__',
    '__license__', '__packagename__', '__url__',
    '__version__','AgentGroup', 'SIRSAgents',
    'SISAgents', 'SIAgents', 'SIRAgents'
]

from .agent import AgentGroup
from .SI import SIAgents
from .SIR import SIRAgents
from .SIS import SISAgents
from .SIRS import SIRSAgents
from .info import (
    __version__,
    __author__,
    __description__,
    __email__,
    __license__,
    __packagename__,
    __url__
)