"""Public censoring-adjustment choices."""

from __future__ import annotations

from enum import Enum


class CensoringModel(str, Enum):
    """Specify how censoring probabilities are estimated.

    Attributes
    ----------
    JOINT
        Fit one model for the first occurrence of protocol deviation or any
        observed censoring event.
    SEPARATE
        Fit one model for protocol deviation and one model for each observed
        censoring event, then multiply their probability contributions.
    PROTOCOL_ONLY
        Fit only the protocol-deviation model. Observed censoring still ends
        follow-up but does not contribute a probability model.
    TREATMENT_PROBABILITY
        Research-oriented alternative that models the observed
        treatment-initiation process and converts its probabilities to
        protocol-adherence weights. It is not required for the standard CCW
        workflow. Observed censoring processes are modeled separately.
    """

    JOINT = "joint"
    SEPARATE = "separate"
    PROTOCOL_ONLY = "protocol_only"
    TREATMENT_PROBABILITY = "treatment_probability"
