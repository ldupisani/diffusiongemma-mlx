"""Reasoning compiler: coded orchestrators that replace the model's thinking mode.

See the package modules:
  engine   — atomic one-shot (thinking-off) calls to the diffusion model
  patterns — composable "patterns of thinking", run by code
  registry — the growing graph of stored orchestrators
  router   — recognize a query type and dispatch to its orchestrator
  capture  — record a teacher (thinking-on) trace for a query
  derive   — scaffold an orchestrator from a captured trace (manual-assisted)
  eval     — score an orchestrator's fidelity against the teacher
"""
