import csv
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from .models import Task
from .scheduler import CriticalChainScheduler

def export_to_csv(scheduler: CriticalChainScheduler, filename: str = "ccpm_schedule.csv", start_date: Optional[datetime] = None):
    """
    Export the schedule to a CSV file.
    
    Args:
        scheduler: The CriticalChainScheduler instance
        filename: The name of the CSV file to create
        start_date: The project start date (defaults to today if None)
        
    Returns:
        str: The path to the created CSV file
    """
    if not scheduler.scheduled:
        scheduler.schedule()
        
    if start_date is None:
        start_date = datetime.now()
        
    # Sort tasks by start time
    sorted_tasks = sorted(scheduler.tasks.values(), key=lambda t: t.start_time)
    
    # Prepare data for CSV
    rows = []
    for task in sorted_tasks:
        # Calculate actual dates based on working calendar
        task_start_date = calculate_date_from_time(start_date, task.start_time, scheduler.working_days)
        task_end_date = calculate_date_from_time(start_date, task.end_time, scheduler.working_days)
        
        # Determine task type
        if task.is_buffer:
            task_type = f"{task.buffer_type.capitalize()} Buffer"
        elif task.is_critical:
            task_type = "Critical Chain"
        else:
            task_type = "Regular Task"
            
        # Create row
        row = {
            "ID": task.id,
            "Name": task.name,
            "Duration": task.duration,
            "Start Time": task.start_time,
            "End Time": task.end_time,
            "Start Date": task_start_date.strftime("%Y-%m-%d"),
            "End Date": task_end_date.strftime("%Y-%m-%d"),
            "Type": task_type,
            "Predecessors": ", ".join(task.predecessors),
            "Resources": ", ".join(task.resources),
            "Progress": f"{task.progress}%"
        }
        rows.append(row)
        
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            "ID", "Name", "Duration", "Start Time", "End Time", 
            "Start Date", "End Date", "Type", "Predecessors", 
            "Resources", "Progress"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            
    return filename

def calculate_date_from_time(start_date: datetime, time_units: float, working_days: List[int]) -> datetime:
    """
    Calculate the actual date based on time units and working calendar.
    
    Args:
        start_date: The project start date
        time_units: The time units from project start
        working_days: List of working days (0=Monday, 6=Sunday)
        
    Returns:
        datetime: The calculated date
    """
    # Simple implementation - each time unit is one day
    current_date = start_date
    remaining_units = time_units
    
    while remaining_units > 0:
        # Move to next day
        current_date += timedelta(days=1)
        
        # Check if it's a working day
        if current_date.weekday() in working_days:
            remaining_units -= 1
            
    return current_date

def set_working_calendar(scheduler: CriticalChainScheduler, calendar_type: str = "standard", start_date: Optional[datetime] = None):
    """
    Set the working calendar for the scheduler.
    
    Args:
        scheduler: The CriticalChainScheduler instance
        calendar_type: The type of calendar ("standard", "24-7", or "weekend")
        start_date: The project start date (defaults to today if None)
        
    Returns:
        CriticalChainScheduler: The updated scheduler
    """
    if start_date is None:
        start_date = datetime.now()
        
    scheduler.start_date = start_date
    
    if calendar_type == "standard":
        # Monday to Friday, 8 AM to 4 PM
        scheduler.working_days = [0, 1, 2, 3, 4]  # Monday to Friday
        scheduler.working_hours = (8, 16)  # 8 AM to 4 PM
    elif calendar_type == "24-7":
        # All days, all hours
        scheduler.working_days = [0, 1, 2, 3, 4, 5, 6]  # All days
        scheduler.working_hours = (0, 24)  # All hours
    elif calendar_type == "weekend":
        # Include weekends
        scheduler.working_days = [0, 1, 2, 3, 4, 5, 6]  # All days
        scheduler.working_hours = (8, 16)  # 8 AM to 4 PM
    else:
        raise ValueError(f"Unknown calendar type: {calendar_type}")
        
    return scheduler