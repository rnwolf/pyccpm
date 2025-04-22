import sys
import os
from ai_ccpm_vba_to_py import Task, Resource, CriticalChainScheduler, prepare_result_dataframe

def test_collect_task_data():
    """Test that the collect_task_data function includes resources and predecessors."""
    # Create a scheduler
    scheduler = CriticalChainScheduler()

    # Add tasks with resources and predecessors
    task_a = Task(1, "Task A", 5, resources="Resource1,Resource2")
    task_b = Task(2, "Task B", 3, predecessors="1", resources="Resource2")
    task_c = Task(3, "Task C", 4, predecessors="2", resources="Resource3")

    scheduler.add_task(task_a)
    scheduler.add_task(task_b)
    scheduler.add_task(task_c)

    # Schedule
    scheduler.schedule()

    # Get the DataFrame
    df = prepare_result_dataframe(scheduler)

    # Print the DataFrame to see if resources and predecessors are included
    print("\nDataFrame with resources and predecessors:")
    print(df)

    # Save the DataFrame to a CSV file to create a reference file for building tests
    csv_file = "tests/reference/test_collect_task_data.csv"
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"\nSaved DataFrame to {csv_file}")

if __name__ == "__main__":
    test_collect_task_data()
