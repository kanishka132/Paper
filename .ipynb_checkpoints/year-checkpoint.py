from graphviz import Digraph

def create_workflow_diagram():
    # Initialize a directed graph
    dot = Digraph(comment='Workflow Diagram')
    
    # Add nodes (each step in the process)
    dot.node('A', 'Start (Search Page)')
    dot.node('B', 'Enter Keyword in Search Bar')
    dot.node('C', 'Display Search Results (Related Papers)')
    dot.node('D', 'Select Paper from List')
    dot.node('E', 'Show Paper Details & Visualizations')
    dot.node('F', 'Click Upload Link')
    dot.node('G', 'Upload PDF File')
    dot.node('H', 'Extract Data from PDF')
    dot.node('I', 'Show Analysis & Visualizations for Uploaded File')
    dot.node('J', 'End')
    
    # Add edges (arrows between steps)
    dot.edge('A', 'B', 'Search for papers')
    dot.edge('B', 'C', 'Keyword Search')
    dot.edge('C', 'D', 'Select Paper')
    dot.edge('D', 'E', 'Show Paper Details')
    dot.edge('A', 'F', 'Use Upload Link')
    dot.edge('F', 'G', 'Upload File')
    dot.edge('G', 'H', 'Extract Data')
    dot.edge('H', 'I', 'Show Analysis & Visualizations')
    dot.edge('I', 'J', 'End Process')
    dot.edge('E', 'J', 'End Process')

    # Render the diagram to a file (in this case, a PNG image)
    dot.render('workflow_diagram', format='png', cleanup=True)

# Create the workflow diagram
create_workflow_diagram()
