from bokeh.io import output_notebook, show
import numpy as np
from scipy.spatial import Voronoi
from bokeh.io import curdoc
from bokeh.layouts import Column, Row, Spacer
from bokeh.models import ColumnDataSource, Slider, Select, Checkbox, Div, GroupBox, Title, LabelSet, Button, SVGIcon
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

noisy_circle_points = np.array([[1.71717172, 3.98989899],
       [3.88888889, 3.13131313],
       [4.04040404, 1.96969697],
       [3.58585859, 1.01010101],
       [1.16161616, 1.21212121],
       [0.65656566, 2.27272727],
       [0.90909091, 3.33333333],
       [2.32323232, 0.55555556],
       [3.08080808, 4.04040404],
       [0.56013424, 3.3210175 ],
       [1.44745819, 3.55306169],
       [2.8893596 , 3.68197514],
       [3.88759903, 3.74643186],
       [3.77668354, 2.07055709],
       [3.20978213, 0.54937845],
       [2.48266945, 1.06503223],
       [1.38583847, 0.72985727],
       [0.73266945, 1.67092541],
       [1.28724692, 1.86429558],
       [1.63231734, 2.8569291 ],
       [2.31013424, 3.48860497],
       [4.18337368, 0.9103361 ],
       [2.43337368, 4.2234116 ],
       [4.49147227, 1.87718692],
       [3.40696523, 1.3486418 ]])

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

triangle_points = np.array([[3.75203565, 3.57884438],
 [1.63231734, 3.63040976],
 [2.54063501, 1.86535189],
 [2.55295896, 2.87087675]])

random_points = np.random.rand(10, 2)*5

#Initilize the global variables
initial_epsilon = 0.3
initial_points = circle_points
complex = Cech_Complex(initial_points)
complex.filter_simplices()
betti = complex.compute_betti(initial_epsilon)

# Define data sources for Bokeh
points_source = ColumnDataSource(data=dict(x=initial_points[:, 0], y=initial_points[:, 1]))
point_labels = ColumnDataSource(data=dict(x=initial_points[:, 0], y=initial_points[:, 1], text=[str(i) for i in range(len(initial_points))]))
ball_source = ColumnDataSource(data=dict(x=initial_points[:, 0], y=initial_points[:, 1], radii=[initial_epsilon]*len(initial_points)))
edge_source = ColumnDataSource(data=dict(x=[initial_points[edge, 0] for edge in complex.get_edges(eps=initial_epsilon)], y=[initial_points[edge, 1] for edge in complex.get_edges(eps=initial_epsilon)]))
triangle_source = ColumnDataSource(data=dict(x=[[[initial_points[triangle, 0]]] for triangle in complex.get_triangles(eps=initial_epsilon)], y=[[[initial_points[triangle, 1]]] for triangle in complex.get_triangles(eps=initial_epsilon)]))
pd_source = complex.get_persistence_pairs()
persistent_pd_source = complex.get_persistence_pairs(eps=initial_epsilon)
for dim in range(2):
    for subdiag in range(3):
        persistent_pd_source[subdiag][dim] = ColumnDataSource(persistent_pd_source[subdiag][dim])
        pd_source[subdiag][dim] = ColumnDataSource(pd_source[subdiag][dim])
voronoi_source = ColumnDataSource(data=dict(x=[], y=[]) )
legend_source = ColumnDataSource(data=dict(x=[0,0], y=[0,0], color=['red', 'blue'], marker=['circle', 'circle'], label=['dim 0', 'dim 1']))

#Define callback functions
def update_pd(eps=None, extended=False):
    '''
    Update the ColumnDataSource for the persistence diagram. 
    If eps is not None, the persistent_pd_source is updated with the persistence pairs up to the given filtration value.
    Else, the pd_source is updated with all persistence pairs.

    Args:
        - eps (float), optional: Filtration value
        - extended (bool), optional: Compute extended persistence
    '''
    if eps:
        pd_pairs = complex.get_persistence_pairs(eps=eps, extended=extended)
        for dim in range(2):
            for subdiag in range(3):
                persistent_pd_source[subdiag][dim].data = dict(x=pd_pairs[subdiag][dim]['x'], y=pd_pairs[subdiag][dim]['y'])
    else:
        pd_pairs = complex.get_persistence_pairs(extended=extended)
        for dim in range(2):
            for subdiag in range(3):
                pd_source[subdiag][dim].data = dict(x=pd_pairs[subdiag][dim]['x'], y=pd_pairs[subdiag][dim]['y'])
    #Update the legend
    if extended:
        legend_source.data = dict(x=[0,0,0,0,0], y=[0,0,0,0,0], color=['red', 'blue', 'grey', 'grey','grey'], marker=['circle', 'circle', 'circle', 'triangle', 'square'], label=['dim 0', 'dim 1', 'ordinary', 'relative', 'essential'])
    else:
        legend_source.data = dict(x=[0,0], y=[0,0], color=['red', 'blue'], marker=['circle', 'circle'], label=['dim 0', 'dim 1'])
    
def update_filtration(eps):
    '''
    Update the simplicial complex and the persistence diagram with the new filtration value.

    Args:
        - eps (float): Filtration value
    '''
    global betti
    ball_source.data['radii'] = [eps]*len(complex.points)
    edge_source.data = dict(x=[complex.points[edge, 0] for edge in complex.get_edges(eps=eps)], y=[complex.points[edge, 1] for edge in complex.get_edges(eps=eps)])
    triangle_source.data = dict(x=[[[complex.points[triangle, 0]]] for triangle in complex.get_triangles(eps=eps)], y=[[[complex.points[triangle, 1]]] for triangle in complex.get_triangles(eps=eps)])
    betti = complex.compute_betti(eps)
    betti_numbers.update(text='<h3>Betti numbers</h3>' + r'<center><p>$$\beta_0 = $$' + f' {betti[0]}' + r'</p>$$\beta_1 = $$'+ f' {betti[1]}')
    #Only uptdate the interactive part of the persistence diagram
    update_pd(eps)

def update_complex(new_filtration, points, extended=False):
    '''
    Update the simplicial complex with the new filtration type.

    Args:
        - new_filtration (str): New complex type
        - points (np.array): New points for the simplicial complex
    '''
    global complex
    if new_filtration == "Cech":
        complex = Cech_Complex(points)
        voronoi_source.data = dict(x=[], y=[])
    elif new_filtration == "Vietoris-Rips":
        complex = VR_Complex(points)
        voronoi_source.data = dict(x=[], y=[])
    elif new_filtration == "Alpha":
        complex = Alpha_Complex(points)
        vor_xs, vor_ys = complex.compute_voronoi()
        voronoi_source.data = dict(x=vor_xs, y=vor_ys)
    complex.filter_simplices()
    update_filtration(epsilon_slider.value)
    update_pd(extended=extended)

def change_points(points, overwrite=False, extended=False):
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
    point_labels.data = dict(x=complex.points[:, 0], y=complex.points[:, 1], text=[str(i) for i in range(len(complex.points))])
    ball_source.data = dict(x=complex.points[:, 0], y=complex.points[:, 1], radii=[epsilon_slider.value]*len(complex.points))
    if complex.__class__.__name__ == "Alpha_Complex":
        vor_xs, vor_ys = complex.compute_voronoi()
        voronoi_source.data = dict(x=vor_xs, y=vor_ys)
    update_filtration(epsilon_slider.value)
    update_pd(extended=extended)

# Define the controls for user interaction
epsilon_slider = Slider(start=0, end=4, value=initial_epsilon, step=0.01, title="Epsilon")
epsilon_slider.on_change('value', lambda attr, old, new: update_filtration(new))
complex_dropdown = Select(title="Complex", value="Cech", options=["Cech", "Vietoris-Rips", "Alpha"], sizing_mode="stretch_width")
complex_dropdown.on_change('value', lambda attr, old, new: update_complex(new, complex.points, extended=extended_persistence_checkbox.active))
extended_persistence_checkbox = Checkbox(active=False, label="Extended Persistence")
extended_persistence_checkbox.on_change('active', lambda attr, old, new: update_pd(extended=new))
controls = GroupBox(child=Column(epsilon_slider, complex_dropdown, extended_persistence_checkbox), checkable=False, sizing_mode="stretch_width")

# Initialize Bokeh figures
complex_fig = figure(width=600, height=600,
                     x_range=(-1, 6), y_range=(-1, 6),
                     tools="",
                     toolbar_location=None)
complex_fig.add_layout(Title(text="Simplicial Complex", text_font_size='20px'), 'above')
complex_fig.scatter('x', 'y', source=points_source, size=8, color='blue')
complex_fig.add_layout(LabelSet(x='x', y='y', text='text', level='glyph', x_offset=5, y_offset=5, source=point_labels))
complex_fig.circle('x', 'y', source=ball_source, radius='radii', fill_alpha=0.2, color='red')
complex_fig.multi_line('x', 'y', source=edge_source, line_width=1, color='blue')
complex_fig.multi_polygons('x', 'y', source=triangle_source, fill_color="rgba(255, 247, 0, 0.5)", line_width=1)
complex_fig.multi_line('x', 'y', source=voronoi_source, line_width=1, color='black')

complex_fig.on_event(Tap, lambda event: change_points(np.array([event.x, event.y]), extended=extended_persistence_checkbox.active))

# Persistence diagram
pd_fig = figure(width=400, height=400, x_range=(-0.2, 5), y_range=(-0.2, 5), toolbar_location=None)
pd_fig.add_layout(Title(text="Persistence Diagram", text_font_size='20px'), 'above')
pd_fig.line([-0.2, 5], [-0.2, 5], color='black', line_width=1)
pd_fig.xaxis.axis_label = "Birth"
pd_fig.yaxis.axis_label = "Death"
pd_fig.scatter('x', 'y', source=legend_source, size=0, color='color', marker='marker', legend_field='label')
# Create a point for each interval
colors = ['red', 'blue', 'green', 'orange']
glyphs = ['circle', 'square', 'triangle']


for dim in range(2):
    for subdiag in range(3):
        pd_fig.scatter('x', 'y', source=persistent_pd_source[subdiag][dim] , size=10, color=colors[dim], alpha=1)
        pd_fig.scatter('x', 'y', source=pd_source[subdiag][dim], color=colors[dim], marker=glyphs[subdiag], size=10, alpha=0.4)

# Define the layout of the Bokeh app
header = Div(text=r'<h1><center>Simplicial Filtrations</h1>', align='center')

#Define the persistence diagram plot
persistence_diagram = Column(children=[Spacer(height=50), pd_fig], align='center')

#Define the simplicial complex plot including the Betti numbers and the starting configuration selection
betti_numbers = Div(text= '<h3>Betti numbers</h3>' + r'<center><p>$$\beta_0 = $$' + f' {betti[0]}' + r'</p>$$\beta_1 = $$'+ f' {betti[1]}', align='center')
start_config_select = Select(title="Starting configuration", 
                             value="Circle", 
                             options=["Circle","Noisy circle", "Two circles", "Triangle" ,"Random", "Empty"], 
                             sizing_mode="stretch_width")
start_config_select.on_change('value', 
                              lambda attr, old, new: change_points(circle_points if new == "Circle" 
                                                                   else noisy_circle_points if new =="Noisy circle" 
                                                                   else two_circles_points if new == "Two circles" 
                                                                   else triangle_points if new == "Triangle" 
                                                                   else np.random.rand(10,2)*5 if new == "Random" 
                                                                   else np.empty(shape=[0,2]), 
                                                                   overwrite=True, 
                                                                   extended=extended_persistence_checkbox.active))
repeat_button = Button(icon=SVGIcon(svg='<svg width="800px" height="800px" viewBox="0 0 24 20" role="img" xmlns="http://www.w3.org/2000/svg" aria-labelledby="repeatIconTitle" stroke="#000000" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none" color="#000000"> <title id="repeatIconTitle">Repeat</title> <path d="M2 13.0399V11C2 7.68629 4.68629 5 8 5H21V5"/> <path d="M19 2L22 5L19 8"/> <path d="M22 9.98004V12.02C22 15.3337 19.3137 18.02 16 18.02H3V18.02"/> <path d="M5 21L2 18L5 15"/> </svg>'), 
                       label='',
                       align='end')
repeat_button.on_click(lambda: change_points(circle_points if start_config_select.value == "Circle" 
                                             else noisy_circle_points if start_config_select.value =="Noisy circle" 
                                             else two_circles_points if start_config_select.value == "Two circles" 
                                             else triangle_points if start_config_select.value == "Triangle" 
                                             else np.random.rand(10,2)*5 if start_config_select.value == "Random" 
                                             else np.empty(shape=[0,2]), 
                                             overwrite=True,
                                             extended=extended_persistence_checkbox.active))
simplicial_complex = Row(children=[betti_numbers, Column(complex_fig, Row(start_config_select, repeat_button))], align='center')

# Define the final layout of the Bokeh app
layout = Column(header, GroupBox(child=Row(children=[Spacer(width=170), Column(children=[controls, persistence_diagram]), simplicial_complex], align='center', sizing_mode='stretch_both'), sizing_mode='stretch_both', align='center'), sizing_mode="stretch_both")

# Add the layout to the Bokeh app
curdoc().add_root(layout)
curdoc().title = "Simplicial Filtrations"