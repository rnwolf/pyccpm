import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Any

from .models import Task, Resource

class CriticalChainScheduler:
    def __init__(self):
        self.tasks = {}
        self.resources = {}
        self.critical_chain = []
        self.secondary_chains = []
        self.project_buffer = None
        self.feeding_buffers = []
        self.resource_buffers = []
        self.scheduled = False
        self.start_date = None
        self.working_days = [0, 1, 2, 3, 4]  # Monday to Friday by default
        self.working_hours = (8, 16)  # 8 AM to 4 PM by default

    def add_task(self, task):
        self.tasks[task.id] = task

    def add_resource(self, resource):
        self.resources[resource.id] = resource

    def find_critical_chain(self):
        """Find the critical chain of the project"""
        if not self.tasks:
            return []

        # Find tasks with no predecessors (start tasks)
        start_tasks = [task for task_id, task in self.tasks.items() if not task.predecessors]

        if not start_tasks:
            print("Error: No starting tasks found (tasks without predecessors)")
            return []

        # Find task with longest total duration path
        max_duration = 0
        best_start_task = None

        for task in start_tasks:
            total_duration = self.calculate_path_duration(task)
            if total_duration > max_duration:
                max_duration = total_duration
                best_start_task = task

        if not best_start_task:
            print("Error: Could not determine best starting task")
            return []

        # Build critical chain from best starting task
        self.critical_chain = [best_start_task]
        best_start_task.is_critical = True
        best_start_task.start_time = 0
        best_start_task.end_time = best_start_task.duration
        best_start_task.type = "Critical"  # Set task type for compatibility with original code

        # Add subsequent tasks to critical chain
        current_task = best_start_task
        while True:
            # Find tasks that have the current task as a predecessor
            antecedents = []
            for task_id, task in self.tasks.items():
                if current_task.id in task.predecessors:
                    antecedents.append(task)

            if not antecedents:
                break

            # Find antecedent with longest duration
            next_task = max(antecedents, key=lambda t: t.duration)

            # Add to critical chain
            self.critical_chain.append(next_task)
            next_task.is_critical = True
            next_task.start_time = current_task.end_time
            next_task.end_time = next_task.start_time + next_task.duration
            next_task.type = "Critical"  # Set task type for compatibility with original code

            # Move to next task
            current_task = next_task

        return self.critical_chain

    def find_antecedents_from_list(self, task, task_list):
        """Find tasks in the given list that have this task as a predecessor"""
        return [t for t in task_list if task.id in t.predecessors]

    def calculate_longest_path_from_start(self, task, visited=None):
        """Calculate the longest path duration starting from a given task"""
        if visited is None:
            visited = set()

        if task.id in visited:
            return 0

        visited.add(task.id)

        # Find tasks that have this task as a predecessor
        successors = []
        for task_id, t in self.tasks.items():
            if task.id in t.predecessors:
                successors.append(t)

        if not successors:
            return task.duration

        max_path = 0
        for successor in successors:
            path_length = self.calculate_longest_path_from_start(successor, visited.copy())
            if path_length > max_path:
                max_path = path_length

        return task.duration + max_path

    def calculate_path_duration(self, start_task, end_task=None, visited=None):
        """Calculate the path duration between two tasks or from a single task"""
        # If end_task is None, use the original implementation from ai_ccpm_vba_to_py.py
        if end_task is None:
            return self.calculate_longest_path_from_start(start_task, visited)
        if visited is None:
            visited = set()

        if start_task.id == end_task.id:
            return start_task.duration

        if start_task.id in visited:
            return 0

        visited.add(start_task.id)

        max_duration = 0
        for task_id, task in self.tasks.items():
            if start_task.id in task.predecessors:
                duration = self.calculate_path_duration(task, end_task, visited.copy())
                if duration > 0:
                    max_duration = max(max_duration, duration + start_task.duration)

        return max_duration

    def find_antecedents(self, task):
        return [self.tasks[pred_id] for pred_id in task.predecessors if pred_id in self.tasks]

    def identify_secondary_chains(self):
        """Identify secondary (feeding) chains"""
        if not self.critical_chain:
            self.find_critical_chain()

        self.secondary_chains = []

        # Create a working copy of tasks that we can modify
        working_tasks = list(self.tasks.values())

        # Remove tasks that are already in the critical chain
        for task in self.critical_chain:
            if task in working_tasks:
                working_tasks.remove(task)

        # For each task in the critical chain
        for cc_task in self.critical_chain:
            # Find tasks that depend on this critical chain task
            dependent_tasks = [t for t in working_tasks if cc_task.id in t.predecessors]

            # Process each dependent task
            for dep_task in dependent_tasks:
                # Start a new chain with this task
                chain = [dep_task]
                dep_task.chain = len(self.secondary_chains)  # Mark which chain it belongs to
                dep_task.type = "Secondary"  # Set task type for compatibility with original code
                dep_task.start_time = cc_task.end_time
                dep_task.end_time = dep_task.start_time + dep_task.duration

                # Set start and finish for compatibility with original code
                dep_task.start = dep_task.start_time
                dep_task.finish = dep_task.end_time

                # Remove from working tasks
                working_tasks.remove(dep_task)

                # Check for resource conflicts and resolve them
                self.resolve_resource_conflicts(dep_task)

                # Build the rest of the chain recursively
                self.build_chain_recursively(dep_task, chain)

                if chain:
                    self.secondary_chains.append(chain)

        return self.secondary_chains

    def build_chain_recursively(self, parent_task, chain, working_tasks=None):
        """Build a chain by recursively adding dependent tasks"""
        # Find tasks that have this task as a predecessor
        dependent_tasks = []
        for task_id, task in self.tasks.items():
            if parent_task.id in task.predecessors and not task.is_critical and task not in chain:
                dependent_tasks.append(task)

        # Process each dependent task
        for dep_task in dependent_tasks:
            dep_task.chain = chain[0].chain if chain else None  # Use the same chain ID
            dep_task.start_time = parent_task.end_time
            dep_task.end_time = dep_task.start_time + dep_task.duration
            dep_task.type = "Secondary"  # Set task type for compatibility with original code

            # Add to chain
            chain.append(dep_task)

            # Check for resource conflicts and resolve them
            self.resolve_resource_conflicts(dep_task)

            # Recursively process this task's dependents
            self.build_chain_recursively(dep_task, chain)

    def resolve_resource_conflicts(self, task):
        """Check for and resolve resource conflicts"""
        # Skip if task has no resources
        if not task.resources:
            return

        # Get all scheduled tasks
        all_scheduled_tasks = self.critical_chain + [t for chain in self.secondary_chains for t in chain]

        for existing_task in all_scheduled_tasks:
            # Skip self comparison
            if existing_task.id == task.id:
                continue

            # Check if tasks overlap in time
            time_overlap = (
                task.start_time < existing_task.end_time and 
                task.end_time > existing_task.start_time
            )

            # Check if tasks share resources
            resource_conflict = any(
                r in existing_task.resources for r in task.resources
            )

            if (time_overlap and resource_conflict and 
                task.resources and existing_task.resources):
                # Conflict detected - schedule task after the existing task
                delay = existing_task.end_time - task.start_time
                task.start_time += delay
                task.end_time += delay

                # Update start and finish for compatibility with original code
                task.start = task.start_time
                task.finish = task.end_time

                # Now check if this impacts other tasks and adjust them as needed
                self.adjust_dependent_tasks(task)

    def adjust_dependent_tasks(self, moved_task):
        """Adjust all tasks that depend on the moved task"""
        # Find all scheduled tasks
        all_scheduled_tasks = self.critical_chain + [
            t for chain in self.secondary_chains for t in chain
        ]

        # Find tasks that depend on the moved task
        dependent_tasks = [
            t for t in all_scheduled_tasks if moved_task.id in t.predecessors
        ]

        for dep_task in dependent_tasks:
            # Only adjust if necessary
            if dep_task.start_time < moved_task.end_time:
                delay = moved_task.end_time - dep_task.start_time
                dep_task.start_time += delay
                dep_task.end_time += delay

                # Update start and finish for compatibility with original code
                dep_task.start = dep_task.start_time
                dep_task.finish = dep_task.end_time

                # Recursively adjust dependent tasks
                self.adjust_dependent_tasks(dep_task)

    def generate_buffers(self):
        """Generate project buffer and feeding buffers"""
        if not self.critical_chain:
            self.find_critical_chain()

        if not self.secondary_chains:
            self.identify_secondary_chains()

        # Project buffer based on critical chain
        if not self.critical_chain:
            print("Error: No critical chain to buffer")
            return

        # Calculate buffer size based on the difference between nominal and actual durations
        buffer_size = sum(
            (t.nominal_duration - t.duration) for t in self.critical_chain
        )

        # If buffer size is too small, use 50% of critical chain duration as a fallback
        if buffer_size < 0.1:
            cc_duration = sum(task.duration for task in self.critical_chain)
            buffer_size = cc_duration * 0.5

        # Create project buffer
        last_task = self.critical_chain[-1]

        # Generate a unique ID for the project buffer
        buffer_id = f"PB_{last_task.id}"

        project_buffer = Task(
            id=buffer_id,
            name="Project Buffer",
            duration=buffer_size,
            nominal_duration=buffer_size,
            predecessors=last_task.id
        )
        project_buffer.is_buffer = True
        project_buffer.buffer_type = "project"
        project_buffer.type = "Buffer"  # Set task type for compatibility with original code
        project_buffer.start_time = last_task.end_time
        project_buffer.end_time = project_buffer.start_time + project_buffer.duration

        # Set start and finish for compatibility with original code
        project_buffer.start = project_buffer.start_time
        project_buffer.finish = project_buffer.end_time

        self.tasks[project_buffer.id] = project_buffer
        self.project_buffer = project_buffer
        self.critical_chain.append(project_buffer)

        # Feeding buffers for each secondary chain
        feeding_buffers = []
        for i, chain in enumerate(self.secondary_chains):
            if not chain:
                continue

            # Calculate feeding buffer size based on the difference between nominal and actual durations
            buffer_size = sum((t.nominal_duration - t.duration) for t in chain)

            # If buffer size is too small, use 50% of chain duration as a fallback
            if buffer_size < 0.1:
                chain_duration = sum(task.duration for task in chain)
                buffer_size = chain_duration * 0.5

            # Generate a unique ID for the feeding buffer
            buffer_id = f"FB_{i+1}"

            # Create feeding buffer
            feeding_buffer = Task(
                id=buffer_id,
                name=f"Feeding Buffer {i+1}",
                duration=buffer_size,
                nominal_duration=buffer_size,
                predecessors=chain[-1].id
            )
            feeding_buffer.is_buffer = True
            feeding_buffer.buffer_type = "feeding"
            feeding_buffer.type = "Buffer"  # Set task type for compatibility with original code
            feeding_buffer.start_time = chain[-1].end_time
            feeding_buffer.end_time = feeding_buffer.start_time + feeding_buffer.duration

            # Set start and finish for compatibility with original code
            feeding_buffer.start = feeding_buffer.start_time
            feeding_buffer.finish = feeding_buffer.end_time

            # Add to tasks dictionary
            self.tasks[feeding_buffer.id] = feeding_buffer
            feeding_buffers.append(feeding_buffer)

            # Find the task in the critical chain that this chain feeds into
            for cc_task in self.critical_chain:
                for task in chain:
                    if task.id in cc_task.predecessors:
                        # Check if feeding buffer would overlap with critical chain task
                        if feeding_buffer.end_time > cc_task.start_time:
                            # Need to move the feeding buffer earlier
                            move_amount = feeding_buffer.end_time - cc_task.start_time
                            feeding_buffer.start_time -= move_amount
                            feeding_buffer.end_time -= move_amount
                            feeding_buffer.start = feeding_buffer.start_time
                            feeding_buffer.finish = feeding_buffer.end_time

                            # This might require moving the entire chain earlier
                            for chain_task in reversed(chain):
                                chain_task.start_time -= move_amount
                                chain_task.end_time -= move_amount
                                chain_task.start = chain_task.start_time
                                chain_task.finish = chain_task.end_time

                        # Update critical task to depend on the buffer instead
                        cc_task.predecessors = [
                            pred for pred in cc_task.predecessors if pred != task.id
                        ]
                        cc_task.predecessors.append(feeding_buffer.id)
                        break

        self.feeding_buffers = feeding_buffers
        return project_buffer, feeding_buffers

    def schedule_remaining_tasks(self):
        """Schedule any remaining tasks"""
        # Create a working copy of tasks that we can modify
        working_tasks = list(self.tasks.values())

        # Remove tasks that are already in the critical chain or secondary chains
        for task in self.critical_chain:
            if task in working_tasks:
                working_tasks.remove(task)

        for chain in self.secondary_chains:
            for task in chain:
                if task in working_tasks:
                    working_tasks.remove(task)

        while working_tasks:
            # Choose the task with fewest predecessors first
            task = min(working_tasks, key=lambda t: len(t.predecessors))

            # Get all scheduled tasks
            all_scheduled_tasks = self.critical_chain + [
                t for chain in self.secondary_chains for t in chain
            ]

            # Find earliest start time based on predecessors
            earliest_start = 0
            for pred_id in task.predecessors:
                pred = next(
                    (t for t in all_scheduled_tasks if str(t.id) == pred_id), None
                )
                if pred:
                    earliest_start = max(earliest_start, pred.end_time)

            # Schedule the task
            task.type = "Free"  # Free task
            task.start_time = earliest_start
            task.end_time = task.start_time + task.duration

            # Set start and finish for compatibility with original code
            task.start = task.start_time
            task.finish = task.end_time

            # Check for resource conflicts
            self.resolve_resource_conflicts(task)

            # Create a new chain for this "free" task
            free_chain = [task]
            self.build_chain_recursively(task, free_chain)

            if free_chain:
                self.secondary_chains.append(free_chain)

            # Remove from working tasks
            working_tasks.remove(task)

    def schedule(self):
        """Run the complete scheduling process"""
        # Reset all task timings
        for task in self.tasks.values():
            task.start_time = 0
            task.end_time = task.duration
            task.is_critical = False
            task.chain = None
            task.type = ""  # Reset task type for compatibility with original code

        # Make a copy of the original task list
        original_tasks = dict(self.tasks)

        # Find the critical chain
        self.find_critical_chain()

        # Identify and schedule secondary chains
        self.identify_secondary_chains()

        # Schedule remaining tasks
        self.schedule_remaining_tasks()

        # Generate buffers
        self.generate_buffers()

        # Update all_scheduled_tasks
        self.all_scheduled_tasks = self.critical_chain + [
            t for chain in self.secondary_chains for t in chain
        ]

        # Reset tasks to the original list for possible rescheduling
        self.tasks = original_tasks

        self.scheduled = True

        # Return all_scheduled_tasks for compatibility with original code
        return self.all_scheduled_tasks

    def visualize(self, show_resources=True):
        """Create a Gantt chart visualization"""
        if not self.scheduled:
            self.schedule()

        # Get all tasks from critical chain and secondary chains
        all_tasks = self.critical_chain + [
            t for chain in self.secondary_chains for t in chain
        ]

        # Sort by start time
        all_tasks.sort(key=lambda t: t.start_time)

        # Define colors based on task type
        colors = {
            "Critical": "red",  # Critical chain
            "Secondary": "green",  # Secondary chain
            "Free": "blue",  # Free task
            "Buffer": "gray",  # Buffer
        }

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 0.5 * len(all_tasks)))

        # Prepare y-axis labels
        y_labels = []
        for i, task in enumerate(all_tasks):
            # Create task label
            label = f"{task.id}: {task.name}"
            if show_resources and task.resources:
                label += f" [{','.join(task.resources)}]"
            y_labels.append(label)

            # Draw bar for task
            ax.barh(
                i,
                task.end_time - task.start_time,
                left=task.start_time,
                color=colors.get(task.type, "purple"),
                edgecolor="black",
            )

            # Add task ID to the bar
            ax.text(
                task.start_time + (task.end_time - task.start_time) / 2,
                i,
                f"{task.id}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

            # Add resource information to the right of the task bar
            if task.resources:
                resource_text = ", ".join([f"{r}:1.0" for r in task.resources])
                ax.text(
                    task.end_time + 0.1,  # Position slightly to the right of the bar
                    i,
                    f"Resources: {resource_text}",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="black",
                )

            # Draw progress if any
            if hasattr(task, "progress") and task.progress > 0:
                progress_width = (task.end_time - task.start_time) * (task.progress / 100)
                ax.barh(i, progress_width, left=task.start_time, color="darkgray", alpha=0.5)

        # Add dependencies as arrows
        for task in all_tasks:
            for pred_id in task.predecessors:
                pred = next((t for t in all_tasks if str(t.id) == pred_id), None)
                if pred:
                    # Find positions
                    pred_idx = all_tasks.index(pred)
                    task_idx = all_tasks.index(task)

                    # Draw arrow from pred end to task start
                    ax.annotate(
                        "",
                        xy=(task.start_time, task_idx),
                        xytext=(pred.end_time, pred_idx),
                        arrowprops=dict(arrowstyle="->", color="black"),
                    )

        # Set y-axis ticks and labels
        ax.set_yticks(range(len(all_tasks)))
        ax.set_yticklabels(y_labels)

        # Set x-axis label and title
        ax.set_xlabel("Time (days)")
        ax.set_title("Critical Chain Project Schedule")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", edgecolor="black", label="Critical Chain"),
            Patch(facecolor="green", edgecolor="black", label="Secondary Chain"),
            Patch(facecolor="blue", edgecolor="black", label="Free Task"),
            Patch(facecolor="gray", edgecolor="black", label="Buffer"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Show resource usage if requested
        if show_resources and self.resources:
            self.visualize_resource()

        plt.tight_layout()

        # For compatibility with manual_tests.py, return plt instead of showing directly
        return plt

    def visualize_resource(self):
        if not self.scheduled:
            self.schedule()

        # Create a new figure for resource visualization
        plt.figure(figsize=(12, 8))

        # Get all resources
        resources = list(self.resources.values())

        # Calculate resource assignments
        for resource in resources:
            resource.assignments = []

        for task in self.tasks.values():
            for resource_id in task.resources:
                if resource_id in self.resources:
                    self.resources[resource_id].assignments.append({
                        'task': task,
                        'start': task.start_time,
                        'end': task.end_time
                    })

        # Sort resources by number of assignments
        resources.sort(key=lambda r: len(r.assignments), reverse=True)

        # Calculate the project duration
        project_end = max(task.end_time for task in self.tasks.values())

        # Set up the plot
        ax = plt.gca()
        ax.set_xlim(0, project_end * 1.05)  # Add some padding
        ax.set_ylim(0, len(resources) * 1.5)

        # Plot each resource's assignments
        for i, resource in enumerate(resources):
            y_pos = len(resources) - i

            # Sort assignments by start time
            resource.assignments.sort(key=lambda a: a['start'])

            for assignment in resource.assignments:
                task = assignment['task']
                start = assignment['start']
                duration = assignment['end'] - assignment['start']

                # Determine color based on task type
                if task.is_buffer:
                    if task.buffer_type == "project":
                        color = 'red'
                    else:
                        color = 'orange'
                elif task.is_critical:
                    color = 'green'
                else:
                    color = 'blue'

                # Plot the assignment bar
                ax.barh(
                    y_pos, 
                    duration, 
                    left=start, 
                    height=0.5, 
                    color=color, 
                    alpha=0.6
                )

                # Add task name
                ax.text(
                    start + duration / 2, 
                    y_pos, 
                    task.name,
                    ha='center', 
                    va='center', 
                    color='black', 
                    fontweight='bold'
                )

        # Set y-tick labels to resource names - FIX HERE
        resource_names = [resource.name for resource in resources]
        # Use set_ticks and then set_ticklabels - this avoids the formatter issue
        y_positions = [len(resources) - i for i in range(len(resources))]
        ax.set_yticks(y_positions)

        # Import the necessary formatter
        from matplotlib.ticker import FixedFormatter
        formatter = FixedFormatter(resource_names)
        ax.yaxis.set_major_formatter(formatter)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label='Critical Chain Task'),
            Patch(facecolor='blue', alpha=0.6, label='Non-Critical Task'),
            Patch(facecolor='red', alpha=0.6, label='Project Buffer'),
            Patch(facecolor='orange', alpha=0.6, label='Feeding Buffer')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Resources')
        ax.set_title('Resource Utilization')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        return plt

    def update_progress(self, task_id, progress_percentage):
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        # Update progress (0-100%)
        task.progress = min(100, max(0, progress_percentage))

        # If this is the first update, set actual start time
        if task.progress > 0 and task.actual_start is None:
            task.actual_start = datetime.now()

        # If task is complete, set actual end time
        if task.progress == 100 and task.actual_end is None:
            task.actual_end = datetime.now()

        # Update buffer consumption
        self._update_buffer_consumption()

        return task

    def _update_buffer_consumption(self):
        # Skip if no critical chain or project buffer
        if not self.critical_chain or not self.project_buffer:
            return

        # Calculate expected progress along the critical chain
        total_cc_duration = sum(task.duration for task in self.critical_chain)
        if total_cc_duration == 0:
            return

        # Calculate actual progress
        completed_duration = 0
        for task in self.critical_chain:
            if task.progress == 100:
                completed_duration += task.duration
            elif task.progress > 0:
                # Partially completed task
                completed_duration += (task.duration * task.progress / 100)

        # Calculate expected progress based on elapsed time
        if self.start_date:
            elapsed_time = (datetime.now() - self.start_date).total_seconds() / 86400  # days
            expected_progress = elapsed_time / total_cc_duration
            expected_completed = expected_progress * total_cc_duration

            # Calculate buffer consumption
            buffer_consumption = max(0, expected_completed - completed_duration)

            # Update project buffer
            self.project_buffer.progress = min(100, buffer_consumption / self.project_buffer.duration * 100)

    def generate_fever_chart(self):
        if not self.critical_chain or not self.project_buffer:
            raise ValueError("Critical chain and project buffer must be generated first")

        # Calculate total critical chain duration
        cc_duration = sum(task.duration for task in self.critical_chain)

        # Calculate completion percentage of critical chain
        completed_duration = 0
        for task in self.critical_chain:
            if task.progress == 100:
                completed_duration += task.duration
            elif task.progress > 0:
                # Partially completed task
                completed_duration += (task.duration * task.progress / 100)

        cc_completion = completed_duration / cc_duration if cc_duration > 0 else 0

        # Calculate buffer consumption
        buffer_consumption = self.project_buffer.progress / 100

        # Create fever chart
        plt.figure(figsize=(10, 8))

        # Define regions
        x = np.linspace(0, 1, 100)
        y_green = x
        y_yellow = x * 1.5
        y_red = np.ones_like(x)

        # Plot regions
        plt.fill_between(x, 0, y_green, color='green', alpha=0.3)
        plt.fill_between(x, y_green, y_yellow, color='yellow', alpha=0.3)
        plt.fill_between(x, y_yellow, y_red, color='red', alpha=0.3)

        # Plot current status
        plt.scatter(cc_completion, buffer_consumption, color='blue', s=100, zorder=5)

        # Add labels and title
        plt.xlabel('Critical Chain Completion (%)')
        plt.ylabel('Project Buffer Consumption (%)')
        plt.title('Project Fever Chart')

        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add percentage labels
        plt.xticks(np.arange(0, 1.1, 0.1), [f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.1)])
        plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(y*100)}%" for y in np.arange(0, 1.1, 0.1)])

        plt.tight_layout()
        return plt
