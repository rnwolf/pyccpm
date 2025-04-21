import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta

class Task:
    def __init__(self, id, name, duration, nominal_duration=None, predecessors="", resources=""):
        self.id = id
        self.name = name
        self.duration = duration
        self.nominal_duration = nominal_duration if nominal_duration is not None else duration
        # Convert predecessors to string if it's not already a string
        if predecessors and not isinstance(predecessors, str):
            predecessors = str(predecessors)
        self.predecessors = predecessors.split(",") if predecessors else []
        self.predecessors = [p.strip() for p in self.predecessors if p.strip()]
        self.resources = resources.split(",") if resources else []
        self.resources = [r.strip() for r in self.resources if r.strip()]
        self._start_time = 0
        self._end_time = 0
        self.is_critical = False
        self.is_buffer = False
        self.buffer_type = None
        self.chain = None
        self.progress = 0
        self.actual_start = None
        self.actual_end = None
        self.type = ""  # Added for compatibility with original code

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def start(self):
        return self._start_time

    @start.setter
    def start(self, value):
        self._start_time = value

    @property
    def finish(self):
        return self._end_time

    @finish.setter
    def finish(self, value):
        self._end_time = value

    def __repr__(self):
        return f"Task({self.id}, {self.name}, {self.duration})"

class Resource:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.assignments = []

    def __repr__(self):
        return f"Resource({self.id}, {self.name})"
