"""
Global UI state for widget synchronization.

Provides shared state flags that all widgets can access
to coordinate behavior during generation, editing, etc.
"""

# Global flag - True when generation is in progress
generating: bool = False