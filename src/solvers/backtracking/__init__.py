"""Backtracking solver package for the Job Shop Scheduling Problem (JSP).

Uses the team problem representation (src.models) and validates its
answers with the team validator (src.core.validator), exactly like the
genetic engine does.
"""

from src.solvers.backtracking.engine import BacktrackingSolver, run_backtracking

__all__ = ["BacktrackingSolver", "run_backtracking"]
