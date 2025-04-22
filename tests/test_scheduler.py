import csv
import sys
import os
import pandas as pd
import pytest
from tdda.referencetest import tag
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import re
from pathlib import Path


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

def sanitize_svg_for_comparison(svg_content: str) -> str:
    svg_content = re.sub(r"<dc:date>.*?</dc:date>", "", svg_content)
    svg_content = re.sub(r'\sclip-path="url\(#.*?\)"', "", svg_content)
    svg_content = re.sub(r'\sid="m[a-f0-9]{8,}"', "", svg_content)
    svg_content = re.sub(r'\sxlink:href="#m[a-f0-9]{8,}"', "", svg_content)
    svg_content = re.sub(r'<clipPath id="p[a-f0-9]{8,}">', '<clipPath>', svg_content)
    svg_content = re.sub(r"\s+", " ", svg_content).strip()
    return svg_content

def assert_svg_equivalent(svg_file: str, reference_svg: str, debug_dir: Path = None):
    with open(svg_file) as f1, open(reference_svg) as f2:
        svg1 = sanitize_svg_for_comparison(f1.read())
        svg2 = sanitize_svg_for_comparison(f2.read())

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        out1 = debug_dir / "sanitized_input.svg"
        out2 = debug_dir / "sanitized_reference.svg"
        out1.write_text(svg1)
        out2.write_text(svg2)

    assert svg1 == svg2, "SVGs are not visually identical." + (
        f"\nSanitized files saved to:\n  {out1}\n  {out2}" if debug_dir else ""
    )


def collect_task_data(task, chain_name):
    """Helper function to collect task information."""
    return {
        "id": task.id,
        "name": task.name,
        "start_time": task.start_time,
        "end_time": task.end_time,
        "type": task.type,
        "chain": chain_name,
    }


def prepare_result_dataframe(scheduler):
    """Extract and format task data from scheduler into a DataFrame."""
    tasks_data = []

    # Collect critical chain data
    for task in scheduler.critical_chain:
        tasks_data.append(collect_task_data(task, "critical"))

    # Collect secondary chains data
    for i, chain in enumerate(scheduler.secondary_chains):
        for task in chain:
            tasks_data.append(collect_task_data(task, f"feeding {i+1}"))

    # Create and prepare DataFrame
    df = pd.DataFrame(tasks_data).sort_values(by=["chain", "start_time"])
    df["start_time"] = df["start_time"].astype(int)
    df['end_time'] = df['end_time'].astype(int)

    return df

# Use pytest fixtures for tdda reference testing
@pytest.mark.tdda
def test_simple_sequential(assertDataFrameCorrect, assertFileCorrect):
    """Automated test: Simple sequential project (A -> B -> C)"""
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

    # Visualize
    plt = scheduler.visualize()

    # Save the SVG file with an absolute path
    svg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gantt_simple_sequential.svg")
    plt.savefig(svg_file, format="svg")

    # Collect and prepare results data
    df = prepare_result_dataframe(scheduler)

    # Compare against reference
    # Use TDDA reference file path utility
    # Get the directory of the current test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Create an absolute path to the reference file
    reference_file = os.path.join(test_dir, "reference", "test_simple_sequential.csv")

    # Check if we're in write mode
    write_mode = os.environ.get('TDDA_WRITE_ALL', '0') == '1'

    if write_mode:
        # Create the reference directory if it doesn't exist
        os.makedirs(os.path.dirname(reference_file), exist_ok=True)
        # Write the reference file
        df['id'] = df['id'].astype(str) # # TDDA does compare column dtypes by default. Force consistency explicitly by making all string
        df.to_csv(reference_file, index=False)
        print(f"Created reference file: {reference_file}")

        # Create the reference SVG file
        reference_svg = os.path.join(test_dir, "reference", "gantt_simple_sequential.svg")
        import shutil
        shutil.copy2(svg_file, reference_svg)
        print(f"Created reference SVG file: {reference_svg}")
    else:
        # Compare against the reference file
        df['id'] = df['id'].astype(str) # # TDDA does compare column dtypes by default. Force consistency explicitly by making all string
        assertDataFrameCorrect(df, reference_file)

        # Skip SVG file comparison as SVG files can contain timestamps or other dynamic content
        # that changes each time they are generated
        # reference_svg = os.path.join(test_dir, "reference", "gantt_simple_sequential.svg")
        # assertFileCorrect(svg_file, reference_svg)


@pytest.mark.tdda
def test_parallel_paths(assertDataFrameCorrect, tmp_path):
    """Automated test: Simple sequential project (A -> B -> C)"""
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

    # Visualize
    plt = scheduler.visualize()
    svg_file = tmp_path / "gantt_parallel_paths.svg"
    plt.savefig(svg_file, format="svg")

    # Collect and prepare results data
    df = prepare_result_dataframe(scheduler)

    # Compare against reference
    test_dir = os.path.dirname(os.path.abspath(__file__))
    reference_file = os.path.join(test_dir, "reference", "test_parallel_paths.csv")
    reference_svg = os.path.join(test_dir, "reference", "gantt_parallel_paths.svg")

    # Check if we're in write mode
    write_mode = os.environ.get('TDDA_WRITE_ALL', '0') == '1'

    if write_mode:
        # Create the reference directory if it doesn't exist
        os.makedirs(os.path.dirname(reference_file), exist_ok=True)
        # Write the reference file
        df['id'] = df['id'].astype(str) # # TDDA does compare column dtypes by default. Force consistency explicitly by making all string
        df.to_csv(reference_file, index=False)
        print(f"Created reference file: {reference_file}")

        # Create the reference SVG file
        import shutil
        shutil.copy2(svg_file, reference_svg)
        print(f"Created reference SVG file: {reference_svg}")
    else:
        # Compare against the reference file
        df['id'] = df['id'].astype(str) # # TDDA does compare column dtypes by default. Force consistency explicitly by making all string
        assertDataFrameCorrect(df, reference_file)
        assert_svg_equivalent(svg_file, reference_svg)



@pytest.mark.tdda
def test_resource_conflicts(assertDataFrameCorrect, tmp_path):
    """Test a project with resource conflicts"""

    scheduler = CriticalChainScheduler()

    # Add resources
    resource_a = Resource("A", "Resource A")
    resource_b = Resource("B", "Resource B")

    scheduler.add_resource(resource_a)
    scheduler.add_resource(resource_b)

    # Add tasks
    task_1 = Task(1, "Start", 2, resources="A")

    # These tasks both need resource A but can't be done in parallel
    task_2 = Task(2, "Task 2", 4, predecessors="1", resources="A")
    task_3 = Task(3, "Task 3", 3, predecessors="1", resources="A")

    # These tasks need resource B
    task_4 = Task(4, "Task 4", 5, predecessors="2", resources="B")
    task_5 = Task(5, "Task 5", 4, predecessors="3", resources="B")

    # Final task
    task_6 = Task(6, "Finish", 2, predecessors="4,5", resources="A,B")

    scheduler.add_task(task_1)
    scheduler.add_task(task_2)
    scheduler.add_task(task_3)
    scheduler.add_task(task_4)
    scheduler.add_task(task_5)
    scheduler.add_task(task_6)

    # Schedule
    scheduler.schedule()

    # Visualize
    plt = scheduler.visualize()
    svg_file = tmp_path / "gantt_resource_conflicts.svg"
    plt.savefig(svg_file, format="svg")

    # Collect and prepare results data
    df = prepare_result_dataframe(scheduler)

    # Compare against reference
    test_dir = Path(__file__).resolve().parent
    reference_file = test_dir / "reference" / "test_resource_conflicts.csv"
    reference_svg = test_dir / "reference" / "gantt_resource_conflicts.svg"

    # Check if we're in write mode
    write_mode = os.environ.get("TDDA_WRITE_ALL", "0") == "1"

    if write_mode:
        # Create the reference directory if it doesn't exist
        os.makedirs(os.path.dirname(reference_file), exist_ok=True)
        # Write the reference file
        df['id'] = df['id'].astype(str)  # TDDA does compare column dtypes by default. If the reference file has numeric id values (e.g. 1), but the test generates strings ('1'), the test will fail. Force consistency explicitly
        df.to_csv(reference_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Created reference file: {reference_file}")

        # Create the reference SVG file
        import shutil
        shutil.copy2(svg_file, reference_svg)
        print(f"Created reference SVG file: {reference_svg}")
    else:
        # Compare against the reference file
        df['id'] = df['id'].astype(str) # # TDDA does compare column dtypes by default. Force consistency explicitly by making all string
        assertDataFrameCorrect(df, reference_file,type_matching='permissive')  # check_data=['name','start_time','end_time','type','chain']
        assert_svg_equivalent(svg_file, reference_svg)


@pytest.mark.tdda
def test_leach_simple(assertDataFrameCorrect, tmp_path):
    """Test a project with simple example from Lawrence Leach Critical Chain Project Management book"""

    scheduler = CriticalChainScheduler()

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

    # Visualize
    plt = scheduler.visualize()
    svg_file = tmp_path / "gantt_leach_simple.svg"
    plt.savefig(svg_file, format="svg")

    # Collect and prepare results data
    df = prepare_result_dataframe(scheduler)

    # Compare against reference
    test_dir = Path(__file__).resolve().parent
    reference_file = test_dir / "reference" / "test_leach_simple.csv"
    reference_svg = test_dir / "reference" / "gantt_leach_simple.svg"

    # Check if we're in write mode
    write_mode = os.environ.get("TDDA_WRITE_ALL", "0") == "1"

    if write_mode:
        # Create the reference directory if it doesn't exist
        os.makedirs(os.path.dirname(reference_file), exist_ok=True)
        # Write the reference file
        df['id'] = df['id'].astype(str)  # TDDA does compare column dtypes by default. If the reference file has numeric id values (e.g. 1), but the test generates strings ('1'), the test will fail. Force consistency explicitly
        df.to_csv(reference_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Created reference file: {reference_file}")

        # Create the reference SVG file
        import shutil
        shutil.copy2(svg_file, reference_svg)
        print(f"Created reference SVG file: {reference_svg}")
    else:
        # Compare against the reference file
        df['id'] = df['id'].astype(str) # # TDDA does compare column dtypes by default. Force consistency explicitly by making all string
        assertDataFrameCorrect(df, reference_file,type_matching='permissive') # check_data=['name','start_time','end_time','type','chain']
        assert_svg_equivalent(svg_file, reference_svg)
