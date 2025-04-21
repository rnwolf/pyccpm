from pyccpm import Task, Resource, CriticalChainScheduler, export_to_csv, set_working_calendar

def test_simple_sequential():
    """Test a simple sequential project."""
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
    
    # Verify critical chain
    critical_chain = scheduler.critical_chain
    assert len(critical_chain) == 3
    assert critical_chain[0].id == "T1"
    assert critical_chain[1].id == "T2"
    assert critical_chain[2].id == "T3"
    
    # Verify project buffer
    assert scheduler.project_buffer is not None
    assert scheduler.project_buffer.duration == 7.5  # 50% of critical chain (15)
    
    print("Simple sequential test passed!")
    return scheduler

if __name__ == "__main__":
    # Run the test
    scheduler = test_simple_sequential()
    
    # Export to CSV
    export_to_csv(scheduler, "test_schedule.csv")
    
    print("Test completed successfully!")