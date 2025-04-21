import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any

from .models import Task
from .scheduler import CriticalChainScheduler

def extract_critical_chain_path(scheduler: CriticalChainScheduler) -> List[str]:
    """Extract the critical chain path as a list of task IDs."""
    if not scheduler.critical_chain:
        scheduler.find_critical_chain()
    return [task.id for task in scheduler.critical_chain]

def visualize_network(scheduler: CriticalChainScheduler):
    """
    Visualize the project network as a directed graph.
    
    Args:
        scheduler: The CriticalChainScheduler instance
        
    Returns:
        matplotlib.pyplot: The plot object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each task
    for task_id, task in scheduler.tasks.items():
        G.add_node(task_id, label=task.name, duration=task.duration)
        
    # Add edges for dependencies
    for task_id, task in scheduler.tasks.items():
        for pred_id in task.predecessors:
            if pred_id in scheduler.tasks:
                G.add_edge(pred_id, task_id)
                
    # Create positions for nodes using a hierarchical layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Prepare node colors
    node_colors = []
    for node in G.nodes():
        task = scheduler.tasks.get(node)
        if task and task.is_buffer:
            if task.buffer_type == "project":
                node_colors.append('red')
            else:
                node_colors.append('orange')
        elif task and task.is_critical:
            node_colors.append('green')
        else:
            node_colors.append('lightblue')
            
    # Draw the graph
    nx.draw(
        G, 
        pos, 
        with_labels=True,
        node_color=node_colors,
        node_size=1500,
        font_size=10,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        edge_color='gray'
    )
    
    # Add node labels with duration
    node_labels = {
        node: f"{scheduler.tasks[node].name}\n({scheduler.tasks[node].duration})" 
        for node in G.nodes() if node in scheduler.tasks
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Critical Chain'),
        Patch(facecolor='lightblue', label='Non-Critical Task'),
        Patch(facecolor='red', label='Project Buffer'),
        Patch(facecolor='orange', label='Feeding Buffer')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Project Network Diagram')
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    
    return plt