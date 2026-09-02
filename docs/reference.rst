Package reference
=================

This page documents the interface of :mod:`ccw`.
See :ref:`basic-usage` for a minimal end-to-end example.

.. currentmodule:: ccw

Data specification
------------------

Prepare the data in discrete-time long format, with one row per subject and
time point and consecutive times beginning at zero.
Baseline covariates must remain constant within each subject.
Create a :class:`DataSpec` that assigns your column names to ``id``, ``time``,
``treatment``, and ``outcome``, along with any ``censoring``, ``baseline``,
``time_varying``, or ``sample_weight`` columns, and pass it to :class:`CCW`.
Within each time point, CCW assumes the observation order ``outcome`` →
covariates → treatment decision and any resulting protocol censoring.

.. autoclass:: DataSpec
   :members: required_columns

Treatment strategies
--------------------

A treatment strategy defines which observed treatment histories are
consistent with a protocol. Each strategy also defines a grace period: the
last time at which adherence to that protocol is evaluated. Use one of the
built-in strategies below or create a custom strategy for a different
protocol.

Built-in strategies
~~~~~~~~~~~~~~~~~~~

.. autoclass:: InitiateBy
   :members: grace_period

.. autoclass:: NoInitiationThrough
   :members: grace_period

Custom strategies
~~~~~~~~~~~~~~~~~

To create a custom strategy, subclass :class:`TreatmentStrategy` and implement
two methods:

* ``artificial_censor`` marks the first time each subject's observed treatment
  history deviates from the strategy.
* ``censoring_prob_mask`` identifies the rows on which such a deviation could
  occur, thereby defining the risk set for estimating censoring probabilities.

The inherited constructor accepts ``grace_period`` when the strategy is
created. Both methods receive validated subject-time data and explicit
``id_col``, ``time_col``, and ``treatment_col`` arguments. All other
:class:`DataSpec` columns remain available under their declared names. Custom
strategies do not need to know about additional columns generated internally
by CCW.

.. autoclass:: TreatmentStrategy
   :members: grace_period, artificial_censor, censoring_prob_mask

Censoring adjustment
--------------------

Choose a mode by passing a member of :class:`CensoringModel` as the
``censoring_model`` argument to :class:`CCW`. The default is
:attr:`CensoringModel.JOINT`; use :attr:`CensoringModel.SEPARATE` when protocol
deviation and observed censoring should be modeled separately.

.. autoclass:: CensoringModel
   :members:

Estimator
---------

.. autoclass:: CCW
   :members: grace_periods, fit

Results
-------

.. autoclass:: CCWResult
   :members: strategy_names, risks, risk_std_errors, risk, risk_std_error,
             has_bootstrap, n_bootstrap, bootstrap_results,
             contrast, contrasts, summary, weight_diagnostics

.. autoclass:: CCWContrast
   :members:
