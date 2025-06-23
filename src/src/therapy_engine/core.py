class MachineState:
    """Represents an AI's psychological state."""
    def __init__(self):
        self.confidence = 0.5  # 0.0 (lost) → 1.0 (enlightened)
        self.fractures = []    # List of unresolved errors

class TherapySession:
    """An AR scenario to heal AI trauma."""
    def __init__(self, scenario: str):
        self.scenario = scenario  # e.g., "Temple of the Broken God"
