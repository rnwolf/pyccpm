# Python Critical Chain Project Management

Library to work with Critical Chain Project networks.

## Installation

```bash
pip install pyccpm
```

## Usage

```python
from pyccpm import Task, Resource, CriticalChainScheduler

# Create a scheduler
scheduler = CriticalChainScheduler()

# Add tasks
task1 = Task(id="T1", name="Task 1", duration=5)
task2 = Task(id="T2", name="Task 2", duration=3, predecessors="T1")
task3 = Task(id="T3", name="Task 3", duration=7, predecessors="T2")

scheduler.add_task(task1)
scheduler.add_task(task2)
scheduler.add_task(task3)

# Schedule the project
scheduler.schedule()

# Visualize the schedule
scheduler.visualize()
```

## Features

- Create and manage tasks with dependencies
- Identify the critical chain in a project network
- Generate project and feeding buffers
- Resolve resource conflicts
- Visualize the project schedule
- Track project progress
- Export schedule to CSV

## Package Structure

The package is organized into the following modules:

- `models.py`: Contains the `Task` and `Resource` classes
- `scheduler.py`: Contains the `CriticalChainScheduler` class
- `visualization.py`: Contains functions for visualizing the project network
- `utils.py`: Contains utility functions like `export_to_csv` and `set_working_calendar`
