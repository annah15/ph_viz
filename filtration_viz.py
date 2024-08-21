from bokeh.io import output_notebook, show
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import Column, Row
from bokeh.models import ColumnDataSource, Slider, Select, Checkbox, Div, GroupBox, Title
from bokeh.plotting import figure, output_file, show
from bokeh.events import Tap
import gudhi as gd
from complexe import Cech_Complex, VR_Complex, Alpha_Complex

# Define the initial points for the different starting configurations
circle_points =np.array([[1.71717172, 3.98989899],
 [3.88888889, 3.13131313],
 [4.04040404, 1.96969697],
 [3.58585859, 1.01010101],
 [1.16161616, 1.21212121],
 [0.65656566, 2.27272727],
 [0.90909091, 3.33333333],
 [2.32323232, 0.55555556],
 [3.08080808, 4.04040404]])

two_circles_points = np.array([[2.42424242, 2.62626263],
 [3.78787879, 2.67676768],
 [4.6969697,  2.22222222],
 [4.5959596,  1.06060606],
 [3.18181818, 0.2020202 ],
 [1.96969697, 1.26262626],
 [2.97979798, 3.93939394],
 [1.86868687, 5.        ],
 [0.60606061, 4.44444444],
 [0.95959596, 2.92929293]])

random_points = np.array([[1.21212121, 3.53535354],
 [3.83838384, 4.39393939],
 [3.98989899, 2.42424242],
 [1.81818182, 1.06060606],
 [2.57575758, 2.72727273],
 [0.3030303,  2.07070707],
 [4.24242424, 0.35353535]])

#Initilize the global variables
initial_epsilon = 0.9
initial_points = circle_points
complex = Cech_Complex(initial_points)
complex.filter_simplices()
betti = complex.compute_betti(initial_epsilon)

# Define data sources for Bokeh
points_source = ColumnDataSource(data=dict(x=initial_points[:, 0], y=initial_points[:, 1]))
ball_source = ColumnDataSource(data=dict(x=initial_points[:, 0], y=initial_points[:, 1], radii=[initial_epsilon]*len(initial_points)))
edge_source = ColumnDataSource(data=dict(x=[initial_points[edge, 0] for edge in complex.get_edges(eps=initial_epsilon)], y=[initial_points[edge, 1] for edge in complex.get_edges(eps=initial_epsilon)]))
triangle_source = ColumnDataSource(data=dict(x=[[[initial_points[triangle, 0]]] for triangle in complex.get_triangles(eps=initial_epsilon)], y=[[[initial_points[triangle, 1]]] for triangle in complex.get_triangles(eps=initial_epsilon)]))
pd_source = complex.get_persistence_pairs()
persistent_pd_source = complex.get_persistence_pairs(eps=initial_epsilon)

for dim in range(4):
    pd_source[dim] = ColumnDataSource(pd_source[dim])
    persistent_pd_source[dim] = ColumnDataSource(persistent_pd_source[dim])

#Define callback functions
def update_filtration(eps):
    '''
    Update the simplicial complex and the persistence diagram with the new filtration value.

    Args:
        - eps (float): Filtration value
    '''
    ball_source.data['radii'] = [eps]*len(complex.points)
    edge_source.data = dict(x=[complex.points[edge, 0] for edge in complex.get_edges(eps=eps)], y=[complex.points[edge, 1] for edge in complex.get_edges(eps=eps)])
    triangle_source.data = dict(x=[[[complex.points[triangle, 0]]] for triangle in complex.get_triangles(eps=eps)], y=[[[complex.points[triangle, 1]]] for triangle in complex.get_triangles(eps=eps)])
    update_pd(eps)

def update_pd(eps=None):
    '''
    Update the ColumnDataSource for the persistence diagram. 
    If eps is not None, the persistent_pd_source is updated with the persistence pairs up to the given filtration value.
    Else, the pd_source is updated with all persistence pairs.

    Args:
        - eps (float), optional: Filtration value
    '''
    if eps:
        pd_pairs = complex.get_persistence_pairs(eps)
        for dim in range(4):
            persistent_pd_source[dim].data = dict(x=pd_pairs[dim]['x'], y=pd_pairs[dim]['y'])
    else:
        pd_pairs = complex.get_persistence_pairs()
        for dim in range(4):
            pd_source[dim].data = dict(x=pd_pairs[dim]['x'], y=pd_pairs[dim]['y'])

def update_complex(new_filtration, points):
    '''
    Update the simplicial complex with the new filtration type.

    Args:
        - new_filtration (str): New complex type
        - points (np.array): New points for the simplicial complex
    '''
    global complex
    if new_filtration == "Cech":
        complex = Cech_Complex(points)
    elif new_filtration == "Vietoris-Rips":
        complex = VR_Complex(points)
    elif new_filtration == "Alpha":
        complex = Alpha_Complex(points)
    complex.filter_simplices()
    update_filtration(epsilon_slider.value)
    update_pd()

def change_points(points, overwrite=False):
    global complex
    if overwrite:
        complex.points = points
    else:
        #If the clicked point already exists, it should be deleted, else it is added as a new point to the complex
        if (np.round(complex.points, 1)==np.round(points,1)).all(axis=1).any():
            complex.points = np.delete(complex.points, (np.round(complex.points, 1)==np.round(points,1)).all(axis=1), axis=0)
        else:
            complex.points = np.append(complex.points, [points], axis=0)
    complex.filter_simplices()
    points_source.data = dict(x=complex.points[:, 0], y=complex.points[:, 1])
    ball_source.data = dict(x=complex.points[:, 0], y=complex.points[:, 1], radii=[epsilon_slider.value]*len(complex.points))
    update_filtration(epsilon_slider.value)
    update_pd()


# Initialize Bokeh figures
complex_fig = figure(width=600, height=600,
                     x_range=(-1, 6), y_range=(-1, 6),
                     tools="tap")
complex_fig.add_layout(Title(text="Simplicial Complex", text_font_size='20px'), 'above')
complex_fig.scatter('x', 'y', source=points_source, size=8, color='blue')
complex_fig.circle('x', 'y', source=ball_source, radius='radii', fill_alpha=0.2, color='red')
complex_fig.multi_line('x', 'y', source=edge_source, line_width=1, color='blue')
complex_fig.multi_polygons('x', 'y', source=triangle_source, fill_color="rgba(255, 247, 0, 0.5)", line_width=1)

complex_fig.on_event(Tap, lambda event: change_points(np.array([event.x, event.y])))

# Persistence diagram
pd_fig = figure(width=400, height=400, x_range=(-0.2, 5), y_range=(-0.2, 5))
pd_fig.add_layout(Title(text="Persistence Diagram", text_font_size='20px'), 'above')
pd_fig.line([-0.2, 5], [-0.2, 5], color='black', line_width=1)
pd_fig.xaxis.axis_label = "Birth"
pd_fig.yaxis.axis_label = "Death"

# Create a point for each interval
colors = ['red', 'blue', 'green', 'orange']

for dim in range(4):
    pd_fig.scatter('x', 'y', source=pd_source[dim] , size=8, color=colors[dim], alpha=0.4)
    pd_fig.scatter('x', 'y', source=persistent_pd_source[dim] , size=8, color=colors[dim], alpha=1)

# Define the layout of the Bokeh app
header = Div(text=r'<h1><center>Simplicial Filtrations</h1>', align='center')

# Define the controls for user interaction
epsilon_slider = Slider(start=0, end=4, value=initial_epsilon, step=0.01, title="Epsilon")
epsilon_slider.on_change('value', lambda attr, old, new: update_filtration(new))
complex_dropdown = Select(title="Complex", value="Cech", options=["Cech", "Vietoris-Rips", "Alpha"], sizing_mode="stretch_width")
complex_dropdown.on_change('value', lambda attr, old, new: update_complex(new, complex.points))
extended_persistence_checkbox = Checkbox(active=False, label="Extended Persistence")
controls = GroupBox(child=Column(epsilon_slider, complex_dropdown, extended_persistence_checkbox), checkable=False, sizing_mode="stretch_width")

#Define the persistence diagram plot
persistence_diagram = Column(pd_fig)

#Define the simplicial complex plot including the Betti numbers and the starting configuration selection
betti_numbers = Div(text=r'<h2>Betti numbers</h2> <p><center>$$\beta_0 = $${betti[0]}</p> <p>$$\beta_1$${betti[1]}</p><p>$$\beta_2 =$${betti[2]}</p>')
start_config_select = Select(title="Starting configuration", value="Circle", options=["Circle", "Two circles", "Random"])
start_config_select.on_change('value', lambda attr, old, new: change_points(circle_points if new == "Circle" else two_circles_points if new == "Two circles" else random_points, True))
simplicial_complex = Row(children=[betti_numbers, Column(complex_fig, start_config_select)])

# Define the final layout of the Bokeh app
layout = Column(header, GroupBox(child=Row(children=[Column(controls, persistence_diagram), simplicial_complex], align='center'), sizing_mode='stretch_both', align='center'), sizing_mode="stretch_both")

# Add the layout to the Bokeh app
curdoc().add_root(layout)
curdoc().title = "Simplicial Filtrations"