Causal Inference (``wraquant.causal``)
=======================================

Causal inference methods for quantitative finance: difference-in-differences,
synthetic control, inverse probability weighting, and treatment effect
estimation. Use these to measure the causal impact of events, policy changes,
or interventions on financial outcomes.

Quick Example
-------------

.. code-block:: python

   from wraquant.causal import treatment

   # Difference-in-differences: impact of an event on asset returns
   did = treatment.difference_in_differences(
       treated=treated_returns,
       control=control_returns,
       treatment_date="2023-06-01",
   )
   print(f"Treatment effect: {did['effect']:.4f}")
   print(f"p-value: {did['p_value']:.4f}")

   # Synthetic control: construct a counterfactual from control assets
   sc = treatment.synthetic_control(
       treated=treated_returns,
       donors=donor_returns_df,
       treatment_date="2023-06-01",
   )
   print(f"Synthetic treatment effect: {sc['effect']:.4f}")

.. seealso::

   - :doc:`econometrics` -- Event studies and panel data methods
   - :doc:`stats` -- Statistical testing foundations

API Reference
-------------

.. automodule:: wraquant.causal
   :members:
   :undoc-members:
   :show-inheritance:

Treatment Effects
~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.causal.treatment
   :members:

Integrations
~~~~~~~~~~~~

.. automodule:: wraquant.causal.integrations
   :members:
