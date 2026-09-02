clone-censor-weight documentation
=================================

``clone-censor-weight`` provides a compact interface for clone-censor-weight analyses of
discrete-time longitudinal observational data.

.. tip::

   For a concise overview of CCW, including package usage and important
   considerations, see the `tutorial notebook
   <https://github.com/yukiregista/clone-censor-weight/blob/main/examples/ccw_tutorial.ipynb>`_.

Installation
------------

``clone-censor-weight`` requires Python 3.10 or later. You can install the package directly from GitHub:

.. code-block:: console

   pip install git+https://github.com/yukiregista/clone-censor-weight.git

To include the optional simulation and paper-reproduction dependencies, use:

.. code-block:: console

   pip install "clone-censor-weight[research] @ git+https://github.com/yukiregista/clone-censor-weight.git"

.. _basic-usage:

Basic usage
-----------

Suppose ``longitudinal_data`` is a pandas data frame with one row per subject
and discrete time point. Within each time point, CCW assumes the observation
order outcome → covariates → treatment decision and any resulting protocol
censoring. First describe the columns and treatment strategies to compare,
then configure and fit the analysis:

.. code-block:: python

   import ccw

   spec = ccw.DataSpec(
       id="patient_id",
       time="day",
       treatment="treatment_started",
       outcome="outcome",
       baseline=("age",),
       time_varying=("severity",),
   )

   analysis = ccw.CCW(
       spec=spec,
       strategies={
           "control": ccw.NoInitiationThrough(2),
           "intervention": ccw.InitiateBy(2),
       },
       weight_models="C(time) + age + severity",
       followup_end=30,
       estimate_at=30,
       n_bootstrap=500,
       bootstrap_seed=2025,
   )

   result = analysis.fit(longitudinal_data)
   print(result.summary())
   print(result.contrast("intervention", "control"))

Here, ``n_bootstrap=500`` requests 500 subject-level bootstrap refits.
Consequently, ``result.summary()`` includes a ``std_error`` column containing
bootstrap standard errors. Set ``n_bootstrap=0`` (the default) to request point
estimates only.

Without bootstrap results, standard-error accessors return ``None`` and
summary tables omit the ``std_error`` column.

Use :meth:`ccw.CCWResult.weight_diagnostics` to inspect the estimated
censoring weights:

.. code-block:: python

   diagnostic_summary, diagnostic_detail = result.weight_diagnostics(
       patterns=("VAR", "HPREV2"),
       min_m=100,
   )

See the :doc:`reference` for all configuration options and result methods.

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference
