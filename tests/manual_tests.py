"""
Manual tests for the Critical Chain Project Management (CCPM) package.

This file contains a menu-driven interface for running the 14 manual tests
from the original AI CCPM VBA Script. These tests are meant to be run manually
and are excluded from automated testing.

Usage:
    python -m tests.manual_tests
"""

import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta
import csv
import random

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(__file__))
# Add the src directory to the Python path
sys.path.insert(0, os.path.join(package_source_path, "src"))

from pyccpm import (
    Task, 
    Resource, 
    CriticalChainScheduler, 
    visualize_network,
    export_to_csv,
    set_working_calendar
)

def test_simple_sequential():
    """Test a simple sequential project (A -> B -> C)"""
    print("\n=== Test Case: Simple Sequential Project ===")

    scheduler = CriticalChainScheduler()

    # Add tasks
    task_a = Task(id="1", name="Task A", duration=5)
    task_b = Task(id="2", name="Task B", duration=3, predecessors="1")
    task_c = Task(id="3", name="Task C", duration=4, predecessors="2")

    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_task(task_c)

    # Schedule
    scheduler.schedule()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start_time}, Finish: {task.end_time}"
        )

    # Visualize
    plt = scheduler.visualize()
    plt.show()

    return scheduler


def test_parallel_paths():
    """Test a project with parallel paths"""
    print("\n=== Test Case: Project with Parallel Paths ===")

    scheduler = CriticalChainScheduler()

    # Add tasks
    task_a = Task(id="1", name="Start", duration=2)

    # Path 1
    task_b1 = Task(id="2", name="Path1-Task1", duration=4, predecessors="1")
    task_b2 = Task(id="3", name="Path1-Task2", duration=3, predecessors="2")

    # Path 2 (longer duration)
    task_c1 = Task(id="4", name="Path2-Task1", duration=3, predecessors="1")
    task_c2 = Task(id="5", name="Path2-Task2", duration=5, predecessors="4")

    # Final task
    task_d = Task(id="6", name="Finish", duration=2, predecessors="3,5")

    scheduler.add_task(task_a)
    scheduler.add_task(task_b1)
    scheduler.add_task(task_b2)
    scheduler.add_task(task_c1)
    scheduler.add_task(task_c2)
    scheduler.add_task(task_d)

    # Schedule
    scheduler.schedule()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start_time}, Finish: {task.end_time}"
        )

    print("\nSecondary Chains:")
    for i, chain in enumerate(scheduler.secondary_chains):
        print(f"Chain {i+1}:")
        for task in chain:
            print(
                f"  {task.id}: {task.name} - Start: {task.start_time}, Finish: {task.end_time}"
            )

    # Visualize
    plt = scheduler.visualize()
    plt.show()

    return scheduler


def test_resource_conflicts():
    """Test a project with resource conflicts"""
    print("\n=== Test Case: Project with Resource Conflicts ===")

    scheduler = CriticalChainScheduler()

    # Add resources
    resource_a = Resource(id="A", name="Resource A")
    resource_b = Resource(id="B", name="Resource B")

    scheduler.add_resource(resource_a)
    scheduler.add_resource(resource_b)

    # Add tasks
    task_1 = Task(id="1", name="Start", duration=2, resources="A")

    # These tasks both need resource A but can't be done in parallel
    task_2 = Task(id="2", name="Task 2", duration=4, predecessors="1", resources="A")
    task_3 = Task(id="3", name="Task 3", duration=3, predecessors="1", resources="A")

    # These tasks need resource B
    task_4 = Task(id="4", name="Task 4", duration=5, predecessors="2", resources="B")
    task_5 = Task(id="5", name="Task 5", duration=4, predecessors="3", resources="B")

    # Final task
    task_6 = Task(id="6", name="Finish", duration=2, predecessors="4,5", resources="A,B")

    scheduler.add_task(task_1)
    scheduler.add_task(task_2)
    scheduler.add_task(task_3)
    scheduler.add_task(task_4)
    scheduler.add_task(task_5)
    scheduler.add_task(task_6)

    # Schedule
    scheduler.schedule()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start_time}, Finish: {task.end_time}"
        )

    # Visualize
    plt = scheduler.visualize()
    plt.show()

    return scheduler


def test_progress_tracking():
    """Test progress tracking functionality"""
    print("\n=== Test Case: Progress Tracking Example ===")

    scheduler = CriticalChainScheduler()

    # Add tasks
    task_a = Task(id="1", name="Task A", duration=5)
    task_b = Task(id="2", name="Task B", duration=3, predecessors="1")
    task_c = Task(id="3", name="Task C", duration=4, predecessors="2")

    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_task(task_c)

    # Schedule
    scheduler.schedule()

    # Update progress
    scheduler.update_progress("1", 100)  # Task A is complete
    scheduler.update_progress("2", 50)   # Task B is 50% complete

    # Print results
    print("\nProgress Status:")
    for task_id, task in scheduler.tasks.items():
        print(f"  {task.id}: {task.name} - Progress: {task.progress}%")

    # Visualize
    plt = scheduler.visualize()
    plt.show()

    return scheduler


def test_complex_network():
    """Test a complex project network"""
    print("\n=== Test Case: Complex Project Network ===")

    scheduler = CriticalChainScheduler()

    # Add tasks
    task_a = Task(id="1", name="Project Start", duration=1)

    # Design phase
    task_b = Task(id="2", name="Requirements", duration=5, predecessors="1")
    task_c = Task(id="3", name="Architecture", duration=7, predecessors="2")

    # Development phase (multiple parallel tracks)
    task_d1 = Task(id="4", name="UI Development", duration=10, predecessors="3")
    task_d2 = Task(id="5", name="Backend Development", duration=12, predecessors="3")
    task_d3 = Task(id="6", name="Database Design", duration=8, predecessors="3")

    # Integration
    task_e = Task(id="7", name="Integration", duration=6, predecessors="4,5,6")

    # Testing
    task_f1 = Task(id="8", name="Unit Testing", duration=4, predecessors="7")
    task_f2 = Task(id="9", name="System Testing", duration=5, predecessors="7")

    # Deployment
    task_g = Task(id="10", name="Deployment", duration=3, predecessors="8,9")

    # Add all tasks
    for task in [task_a, task_b, task_c, task_d1, task_d2, task_d3, task_e, task_f1, task_f2, task_g]:
        scheduler.add_task(task)

    # Schedule
    scheduler.schedule()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start_time}, Finish: {task.end_time}"
        )

    # Visualize
    plt = scheduler.visualize()
    plt.show()

    return scheduler


def test_manufacturing_project():
    """Test a manufacturing project example"""
    print("\n=== Test Case: Manufacturing Project Example ===")

    scheduler = CriticalChainScheduler()

    # Add resources
    resources = {
        "ENG": Resource(id="ENG", name="Engineering"),
        "MACH": Resource(id="MACH", name="Machining"),
        "ASSM": Resource(id="ASSM", name="Assembly"),
        "QA": Resource(id="QA", name="Quality Assurance"),
        "PACK": Resource(id="PACK", name="Packaging")
    }

    for resource in resources.values():
        scheduler.add_resource(resource)

    # Add tasks
    tasks = [
        Task(id="1", name="Design Product", duration=10, resources="ENG"),
        Task(id="2", name="Create Prototype", duration=5, predecessors="1", resources="ENG,MACH"),
        Task(id="3", name="Test Prototype", duration=3, predecessors="2", resources="ENG,QA"),
        Task(id="4", name="Revise Design", duration=4, predecessors="3", resources="ENG"),
        Task(id="5", name="Order Materials", duration=8, predecessors="4"),
        Task(id="6", name="Receive Materials", duration=3, predecessors="5"),
        Task(id="7", name="Machine Parts A", duration=6, predecessors="6", resources="MACH"),
        Task(id="8", name="Machine Parts B", duration=7, predecessors="6", resources="MACH"),
        Task(id="9", name="Assemble Product", duration=5, predecessors="7,8", resources="ASSM"),
        Task(id="10", name="Quality Check", duration=2, predecessors="9", resources="QA"),
        Task(id="11", name="Packaging", duration=1, predecessors="10", resources="PACK"),
        Task(id="12", name="Ship to Customer", duration=2, predecessors="11")
    ]

    for task in tasks:
        scheduler.add_task(task)

    # Schedule
    scheduler.schedule()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start_time}, Finish: {task.end_time}"
        )

    # Visualize
    plt = scheduler.visualize(show_resources=True)
    plt.show()

    return scheduler


def test_software_development():
    """Test a software development project"""
    print("\n=== Test Case: Software Development Project ===")

    scheduler = CriticalChainScheduler()

    # Add resources
    resources = {
        "PM": Resource(id="PM", name="Project Manager"),
        "BA": Resource(id="BA", name="Business Analyst"),
        "DEV": Resource(id="DEV", name="Developer"),
        "QA": Resource(id="QA", name="QA Engineer"),
        "OPS": Resource(id="OPS", name="Operations")
    }

    for resource in resources.values():
        scheduler.add_resource(resource)

    # Add tasks
    tasks = [
        Task(id="1", name="Project Kickoff", duration=1, resources="PM,BA"),
        Task(id="2", name="Requirements Gathering", duration=5, predecessors="1", resources="BA"),
        Task(id="3", name="System Design", duration=7, predecessors="2", resources="DEV"),
        Task(id="4", name="Frontend Development", duration=10, predecessors="3", resources="DEV"),
        Task(id="5", name="Backend Development", duration=12, predecessors="3", resources="DEV"),
        Task(id="6", name="Database Development", duration=8, predecessors="3", resources="DEV"),
        Task(id="7", name="Integration", duration=5, predecessors="4,5,6", resources="DEV"),
        Task(id="8", name="Testing", duration=8, predecessors="7", resources="QA"),
        Task(id="9", name="Bug Fixing", duration=5, predecessors="8", resources="DEV"),
        Task(id="10", name="User Acceptance Testing", duration=5, predecessors="9", resources="BA,QA"),
        Task(id="11", name="Deployment", duration=2, predecessors="10", resources="OPS"),
        Task(id="12", name="Training", duration=3, predecessors="11", resources="BA"),
        Task(id="13", name="Go Live", duration=1, predecessors="12", resources="PM,OPS")
    ]

    for task in tasks:
        scheduler.add_task(task)

    # Schedule
    scheduler.schedule()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start_time}, Finish: {task.end_time}"
        )

    # Visualize
    plt = scheduler.visualize(show_resources=True)
    plt.show()

    return scheduler


def test_buffer_sizing_strategies():
    """Test different buffer sizing strategies"""
    print("\n=== Test Case: Buffer Sizing Strategies Comparison ===")

    # Create a project
    scheduler = CriticalChainScheduler()

    # Add tasks
    task_a = Task(id="1", name="Task A", duration=5)
    task_b = Task(id="2", name="Task B", duration=3, predecessors="1")
    task_c = Task(id="3", name="Task C", duration=4, predecessors="2")

    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_task(task_c)

    # Schedule with default buffer sizing (50% of chain)
    scheduler.schedule()

    # Print results
    print("\nDefault Buffer Sizing (50% of chain):")
    print(f"  Critical Chain Duration: {sum(task.duration for task in scheduler.critical_chain)}")
    print(f"  Project Buffer Duration: {scheduler.project_buffer.duration}")
    print(f"  Total Project Duration: {max(task.end_time for task in scheduler.tasks.values())}")

    # Visualize
    plt = scheduler.visualize()
    plt.show()

    return scheduler


def test_export_functionality():
    """Test exporting schedule to CSV"""
    print("\n=== Test Case: Export Schedule to CSV ===")

    # Create a project
    scheduler = CriticalChainScheduler()

    # Add tasks
    task_a = Task(id="1", name="Task A", duration=5)
    task_b = Task(id="2", name="Task B", duration=3, predecessors="1")
    task_c = Task(id="3", name="Task C", duration=4, predecessors="2")

    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_task(task_c)

    # Schedule
    scheduler.schedule()

    # Export to CSV
    filename = "ccpm_schedule.csv"
    export_to_csv(scheduler, filename)

    print(f"\nSchedule exported to {filename}")

    # Read and display the CSV content
    print("\nCSV Content:")
    try:
        with open(filename, 'r') as f:
            for line in f:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")

    return scheduler


def test_monte_carlo_simulation():
    """Test Monte Carlo simulation for project duration"""
    print("\n=== Test Case: Monte Carlo Simulation ===")

    # Create a project
    scheduler = CriticalChainScheduler()

    # Add tasks
    task_a = Task(id="1", name="Task A", duration=5)
    task_b = Task(id="2", name="Task B", duration=3, predecessors="1")
    task_c = Task(id="3", name="Task C", duration=4, predecessors="2")

    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_task(task_c)

    # Schedule
    scheduler.schedule()

    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation (100 iterations)...")

    # Store simulation results
    simulation_results = []

    # Run 100 simulations
    for i in range(100):
        # Create a copy of the scheduler for this simulation
        sim_scheduler = CriticalChainScheduler()

        # Add tasks with random durations (±30%)
        for task_id, task in scheduler.tasks.items():
            # Skip buffer tasks
            if task.is_buffer:
                continue

            # Generate random duration
            variation = random.uniform(0.7, 1.3)
            sim_duration = task.duration * variation

            # Create new task
            sim_task = Task(
                id=task.id,
                name=task.name,
                duration=sim_duration,
                predecessors=",".join(task.predecessors),
                resources=",".join(task.resources)
            )

            sim_scheduler.add_task(sim_task)

        # Schedule
        sim_scheduler.schedule()

        # Get project duration (excluding project buffer)
        project_duration = max(
            task.end_time for task_id, task in sim_scheduler.tasks.items() 
            if not task.is_buffer or task.buffer_type != "project"
        )

        simulation_results.append(project_duration)

    # Calculate statistics
    simulation_results.sort()
    p50 = simulation_results[49]  # 50th percentile
    p80 = simulation_results[79]  # 80th percentile
    p90 = simulation_results[89]  # 90th percentile

    print(f"\nSimulation Results:")
    print(f"  P50 (50% confidence): {p50:.1f} days")
    print(f"  P80 (80% confidence): {p80:.1f} days")
    print(f"  P90 (90% confidence): {p90:.1f} days")

    # Compare with standard CCPM approach
    cc_duration = sum(task.duration for task in scheduler.critical_chain)
    buffer_size = scheduler.project_buffer.duration
    total_duration = cc_duration + buffer_size

    print(f"\nStandard CCPM Approach:")
    print(f"  Critical Chain: {cc_duration} days")
    print(f"  Project Buffer: {buffer_size} days")
    print(f"  Total Duration: {total_duration} days")

    # Calculate what confidence level the standard buffer provides
    buffer_confidence = (
        sum(1 for duration in simulation_results if duration <= total_duration)
        / len(simulation_results)
        * 100
    )
    print(f"  Standard CCPM Buffer Provides: {buffer_confidence:.1f}% confidence")

    # Visualize the results
    plt.figure(figsize=(12, 8))

    # Histogram of simulation results
    plt.hist(
        simulation_results, bins=30, alpha=0.7, color="lightblue", edgecolor="black"
    )

    # Add lines for key percentiles
    plt.axvline(
        p50, color="green", linestyle="--", linewidth=2, label=f"P50: {p50:.1f} days"
    )
    plt.axvline(
        p80, color="orange", linestyle="--", linewidth=2, label=f"P80: {p80:.1f} days"
    )
    plt.axvline(
        p90, color="red", linestyle="--", linewidth=2, label=f"P90: {p90:.1f} days"
    )

    # Add line for standard CCPM
    plt.axvline(
        total_duration,
        color="purple",
        linestyle="-",
        linewidth=2,
        label=f"Standard CCPM: {total_duration} days ({buffer_confidence:.1f}%)",
    )

    plt.xlabel("Project Duration (days)")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Simulation Results for Project Duration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return scheduler


def test_larry_simple():
    """Test a project with resource conflicts"""
    print("\n=== Test Case: Project simple example from Larry's book  ===")

    scheduler = CriticalChainScheduler()

    #resources = ["Red", "Green", "Magenta", "Blue"]
    # Define tasks - ID: Task(ID, Name, Duration, Dependencies, Resources)
    # tasks = {
    #     1: Task(1, "T1.1", 30, [], ["Red"]),
    #     2: Task(2, "T1.2", 20, [1], ["Green"]),
    #     3: Task(3, "T3", 30, [5, 2], ["Magenta"]),
    #     4: Task(4, "T2.1", 20, [], ["Blue"]),
    #     5: Task(5, "T2.2", 10, [4], ["Green"]),
    # }

    # Add resources
    resource_r = Resource("R", "Red")
    resource_g = Resource("G", "Green")
    resource_m = Resource("M", "Magenta")
    resource_b = Resource("B", "Blue")

    scheduler.add_resource(resource_r)
    scheduler.add_resource(resource_g)
    scheduler.add_resource(resource_m)
    scheduler.add_resource(resource_b)

    # Add tasks with aggresive optimistic durations
    task_1 = Task(1, "T1.1", 30, resources="R")
    task_2 = Task(2, "T1.2", 20, predecessors="1", resources="G")

    task_3 = Task(3, "T3", 30, predecessors="2,5", resources="M")

    task_4 = Task(4, "T2.1", 20, resources="B")
    task_5 = Task(5, "T2.2", 10, predecessors="4", resources="G")


    scheduler.add_task(task_1)
    scheduler.add_task(task_2)
    scheduler.add_task(task_3)
    scheduler.add_task(task_4)
    scheduler.add_task(task_5)

    # Schedule
    scheduler.schedule()
    scheduler.visualize_resource()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start}, Finish: {task.finish}, Resources: {task.resources}"
        )

    print("\nSecondary Chains:")
    for i, chain in enumerate(scheduler.secondary_chains):
        print(f"Chain {i+1}:")
        for task in chain:
            print(
                f"  {task.id}: {task.name} - Start: {task.start}, Finish: {task.finish}, Resources: {task.resources}"
            )

    # Visualize
    scheduler.visualize()
    # visualize resource load
    scheduler.visualize_resource()

    return scheduler


def test_larry_complex():
    """Test a project with resource conflicts"""
    print("\n=== Test Case: Project with Resource Conflicts ===")

    scheduler = CriticalChainScheduler()

    # Add resources
    resource_r = Resource("R", "Red")
    resource_g = Resource("G", "Green")
    resource_m = Resource("M", "Magenta")
    resource_b = Resource("B", "Blue")
    resource_k = Resource("K", "Black")

    scheduler.add_resource(resource_r)
    scheduler.add_resource(resource_g)
    scheduler.add_resource(resource_m)
    scheduler.add_resource(resource_b)
    scheduler.add_resource(resource_k)

    # Add A tasks with aggressive, optimistic durations
    task_1 = Task(1, "A-1", 5, resources="M")
    task_2 = Task(2, "A-2", 10, predecessors="1", resources="K")
    task_3 = Task(3, "A-3", 15, predecessors="2", resources="G")
    task_4 = Task(4, "A-4", 10, predecessors="3", resources="R")
    task_5 = Task(5, "A-5", 20, predecessors="4,9", resources="M")
    task_6 = Task(6, "A-6", 15, predecessors="5", resources="R")

    # Add B tasks
    task_7 = Task(7, "B-2", 10, resources="M")
    task_8 = Task(8, "B-3", 10, predecessors="7",resources="B")
    task_9 = Task(9, "B-4", 5, predecessors="8", resources="R")

    # Add C tasks
    task_10 = Task(10, "C-3", 15, resources="M")
    task_11 = Task(11, "C-4", 10, predecessors="10",resources="B")
    task_12 = Task(12, "C-5", 15, predecessors="11,15", resources="R")
    task_13 = Task(13, "C-6", 5, predecessors="12", resources="R")

    # Add D tasks
    task_14 = Task(14, "D-3", 20, resources="B")
    task_15 = Task(15, "D-4", 5, predecessors="10",resources="G")

    # End Task
    task_16 = Task(16, "End", 0, predecessors="13,6")


    scheduler.add_task(task_1)
    scheduler.add_task(task_2)
    scheduler.add_task(task_3)
    scheduler.add_task(task_4)
    scheduler.add_task(task_5)
    scheduler.add_task(task_6)
    scheduler.add_task(task_7)
    scheduler.add_task(task_8)
    scheduler.add_task(task_9)
    scheduler.add_task(task_10)
    scheduler.add_task(task_11)
    scheduler.add_task(task_12)
    scheduler.add_task(task_13)
    scheduler.add_task(task_14)
    scheduler.add_task(task_15)
    scheduler.add_task(task_16)

    # Schedule
    scheduler.schedule()

    # Print results
    print("\nCritical Chain:")
    for task in scheduler.critical_chain:
        print(
            f"  {task.id}: {task.name} - Start: {task.start}, Finish: {task.finish}, Resources: {task.resources}"
        )

    print("\nSecondary Chains:")
    for i, chain in enumerate(scheduler.secondary_chains):
        print(f"Chain {i+1}:")
        for task in chain:
            print(
                f"  {task.id}: {task.name} - Start: {task.start}, Finish: {task.finish}, Resources: {task.resources}"
            )

    # Visualize
    scheduler.visualize()
    # visualize resource load
    scheduler.visualize_resource()

    return scheduler


def main():
    """Run the manual tests."""
    print("========== Critical Chain Project Management Demo ==========")
    print("Select a test to run:")
    print("1. Simple Sequential Project")
    print("2. Project with Parallel Paths")
    print("3. Project with Resource Conflicts")
    print("4. Progress Tracking Example")
    print("5. Complex Project Network")
    print("6. Manufacturing Project Example")
    print("7. Software Development Project")
    print("8. Buffer Sizing Strategies Comparison")
    print("9. Export Schedule to CSV")
    print("10. Monte Carlo Simulation")
    print("11. Network Diagram Visualization")
    print("12. Run All Tests")
    print("13. Larry's Simple Project")
    print("14. Larry's Complex Project")

    try:
        choice = int(input("\nEnter your choice (1-14): "))

        if choice == 1:
            scheduler = test_simple_sequential()
        elif choice == 2:
            scheduler = test_parallel_paths()
        elif choice == 3:
            scheduler = test_resource_conflicts()
        elif choice == 4:
            scheduler = test_progress_tracking()
        elif choice == 5:
            scheduler = test_complex_network()
        elif choice == 6:
            scheduler = test_manufacturing_project()
        elif choice == 7:
            scheduler = test_software_development()
        elif choice == 8:
            scheduler = test_buffer_sizing_strategies()
        elif choice == 9:
            scheduler = test_export_functionality()
        elif choice == 10:
            scheduler = test_monte_carlo_simulation()
        elif choice == 11:
            # Run complex network and visualize
            scheduler = test_complex_network()
            plt = visualize_network(scheduler)
            plt.show()
        elif choice == 12:
            print("\nRunning all tests sequentially...")
            test_simple_sequential()
            test_parallel_paths()
            test_resource_conflicts()
            test_progress_tracking()
            test_complex_network()
            test_manufacturing_project()
            test_software_development()
            test_buffer_sizing_strategies()
            test_export_functionality()
            test_monte_carlo_simulation()
            scheduler = test_complex_network()
            plt = visualize_network(scheduler)
            plt.show()
        elif choice == 13:
            scheduler = test_larry_simple()
        elif choice == 14:
            scheduler = test_larry_complex()
        else:
            print("Invalid choice. Please enter a number between 1 and 14.")
    except ValueError:
        print("Please enter a valid number.")

    print(
        "\nDemo completed. Thank you for exploring Critical Chain Project Management!"
    )
    print("============================================================")

if __name__ == "__main__":
    main()
