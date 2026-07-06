"""The coordinator: ~0.6B encoder + ~10K-parameter head + decision policy.

Modules (filled in against docs/SPEC.md §3):
  encoder.py  -- load the 0.6B LM, extract the hidden-state feature for a query
  head.py     -- the ~10K-param head; the flat parameter vector CMA-ES optimizes
  policy.py   -- map head outputs -> (model, role, stop) decision
"""
