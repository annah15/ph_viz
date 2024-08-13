import numpy as np
import ripser
from itertools import combinations
from complexe import Complex, Cech_Complex, VR_Complex, Alpha_Complex
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
from dash.dependencies import Input, State, Output, ALL, MATCH
import pprint
import math

# Initialize random see, load template and set initial variables to compute complex
np.random.seed(298)
load_figure_template(['darkly'])

initial_epsilon = 0.5
initial_points = []
complex = Cech_Complex(initial_epsilon, initial_points)

# Initialize figure to draw the complex
initial_fig = go.Figure()

# Add trace for balls
initial_fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=.5, color='#c8e4ff'), marker=dict(size=1, opacity=0.5), fill='toself', opacity=0.3, name='balls', hoverinfo='none'))
# Add trace for points
initial_fig.add_trace(go.Scatter(
    x=[],
    y=[],
    mode='markers',
    marker=dict(color='white', size=6),
    name='points',
    ))
#Add trace for edges
initial_fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=1), marker=dict(color='white', size=1), name='edges', hoverinfo='none'))
#Add trace for triangles
#initial_fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(width=0), fill='toself', opacity=0.5, name='triangles', hoverinfo='none'))

# Add invisible points in the background to enable click events with arbitrary x and y values
initial_fig.add_traces(
    go.Scatter(
        x=np.repeat(np.linspace(0, 5, 100), 100), y=np.tile(np.linspace(0, 5, 100), 100),
        marker=dict(color='rgba(0,0,0,0)'),
        name='background',
        hoverinfo='none'
    )
)
# Set layout
initial_fig.update_layout(showlegend=False,
                  template='darkly',
                  autosize=False,
                  width=600,
                  height=600,
                  margin=dict(l=0, r=0, b=0, t=0, pad=0),)               
initial_fig.update_xaxes(title_text='X', range=[-1, 6])
initial_fig.update_yaxes(title_text='Y', range=[-1, 6], scaleanchor='x', scaleratio=1)


#draw_balls(fig, initial_points, initial_epsilon)

#Compute persistence diagram with Ripser
#diagrams = ripser.ripser(complex.points)['dgms']
#y_values = [pt[1] if pt[1] != np.inf else 0 for pt in diagrams[0]] + [pt[1] for pt in diagrams[1]]
# Replace infinity with max value in all diagrams
#diagrams = [np.array([[pt[0], pt[1] if pt[1] != np.inf else np.max(y_values) + .2] for pt in diagram]) for diagram in diagrams]
pd_fig = go.Figure()
# pd_fig.add_trace(go.Scatter(x=[pt[0] for pt in diagrams[0]], y=[pt[1] for pt in diagrams[0]], mode='markers', marker=dict(size=10), name='0D Homology'))
# pd_fig.add_trace(go.Scatter(x=[pt[0] for pt in diagrams[1]], y=[pt[1] for pt in diagrams[1]], mode='markers', marker=dict(size=10), name='1D Homology'))
# #Add diagonal
# pd_fig.add_trace(go.Scatter(x=[-0.2, 2.5], y=[-0.2, 2.5], mode='lines', line=dict(color='white', width=1), name='diagonal', hoverinfo="none", showlegend=False))
# pd_fig.update_layout(showlegend=True, template='darkly', xaxis_title='Birth', yaxis_title='Death', xaxis_range=[-0.2, 2.5], yaxis_range=[-0.2, 2.5])


# Setup Dash app
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME, dbc_css])

header = html.H4(
    "Filtration of a Pointcloud", className="bg-primary text-white p-2 mb-2 text-center"
)

slider = html.Div(
    [
        dbc.Label("Epsilon"),
        dcc.Slider(
            0,
            2,
            0.1,
            updatemode="drag",
            id="epsilon-slider",
            value=initial_epsilon,
            tooltip={"placement": "bottom", "always_visible": True},
            className="p-0",
        ),
    ], style={'margin-top': '10px', 'margin-bottom': '30px'},
    className="mb-4",
)

filtration_select = html.Div(
    [
        dbc.Label("Select filtration"),
        dcc.Dropdown(
            ["Cech", "Alpha", "Vietoris-Rips"],
            "Cech",
            id="complex",
            clearable=False,
        ),
    ],
    className="mb-4",
)

start_config_select = html.Div(
    [
        dbc.Label("Select starting configuration"),
        dcc.Dropdown(
            ["Circle", "Two Circles", "Random"],
            "Circle",
            id="start_config",
            clearable=False,
        ),
    ],
    className="mb-4",
)

controls = dbc.Card(
    [filtration_select, slider, start_config_select],
    body=True,
)

persistence_diagram = dbc.Card(
    [dcc.Graph(figure=pd_fig, id='persistence-diagram', className='m-4 dbc dbc-ag-grid')],
    body=True,
)

betti_numbers= html.Div([
    dcc.Markdown(f'''$$\\beta_0 = 20$$''',mathjax=True),
    dcc.Markdown(f'''$$\\beta_1 = 0$$''',mathjax=True),], id='betti',
    style={'float': 'right','margin': 'auto'}
    )

graphs = dbc.Card(
    [dbc.Row([
        betti_numbers,
        dcc.Graph(figure=initial_fig, id='complex_fig', className='m-4 dbc dbc-ag-grid', style={ 'float': 'right','margin': 'auto'}),]
        , style={'justify-items': 'center', "align-items": "center"}),
        
    ],
    body=True,
    style={"justify-content": "center", "align-items": "center"},
)

app.layout = html.Div([
    dcc.Store(id='points', data=initial_points),
    dbc.Container(
    [
        header,
        dbc.Row([
            dbc.Col([
                controls,
                persistence_diagram
            ],  width=5),
            dbc.Col([
                graphs
            ], width=7),
        ],
        align="center",

        ),
    ],
    fluid=True,
    className="dbc dbc-ag-grid",
)])

@app.callback(
    Output("complex_fig", "figure"),
    Output("betti", "children"),
    Output("points", "data"),
    Input("complex_fig", "clickData"),
    Input("complex", "value"),
    Input("epsilon-slider", "value"),
    State("points", "data"),
    State("complex_fig", "figure")
)
def update_complex(new_point, complex_type, eps, points, fig):
    fig = go.Figure(initial_fig)
    #If points is empty it has to be initialize as 2d np.array, otherwise it has to be casted to a np.array
    if len(points) <1:
        points = np.empty(shape=[0,2])
    else:
        points = np.array(points)
    #If update has been triggered by a click event, a new point has to be added or an existing point has to be deleted
    if dash.callback_context.triggered_id == "complex_fig":
        x, y = new_point['points'][0]['x'], new_point['points'][0]['y']
        #If the clicked point already exists, it should be deleted, else it is added as a new point to the complex
        if (points==[x,y]).all(axis=1).any():
            points = np.delete(points, (points==[x,y]).all(axis=1), axis=0)
        else:
            points = np.append(points, [[x, y]], axis=0)
    #If the update has been triggered by a change of the simplicial filtration, a new complex has to be initialized
    if complex_type == "Cech":
        complex = Cech_Complex(eps, points)
    elif complex_type == "Vietoris-Rips":
        complex = VR_Complex(eps, points)
    else:
        complex = Alpha_Complex(eps, points)
    #Now that the filtration with respect to the filtration eps is computed, draw the components
    if len(points) > 0:
        fig.update_traces(x=points[:, 0], y=points[:, 1], selector=dict(name='points'), overwrite=False)
    # draw_points(complex_fig, points)

    x_ball , y_ball = draw_balls(points, eps)
    fig.update_traces(x=x_ball, 
                      y=y_ball, 
                      selector=dict(name='balls'),
                      overwrite=False)

    x_edge,y_edge = draw_one_simplices(points, complex.edges)
    fig.update_traces( x=x_edge,
                       y=y_edge,
                       selector=dict(name='edges'),
                       overwrite=True)
    
    triangle_shapes = draw_two_simplices(points, complex.triangles)
    fig.update_layout(shapes=triangle_shapes, overwrite=True)
    print(fig.layout)
    return fig, [dcc.Markdown(f'''$$\\beta_0 = {complex.betti0}$$''',mathjax=True),dcc.Markdown(f'''$$\\beta_1 ={complex.betti1}$$''',mathjax=True)], points

def draw_points(fig:go.Figure, points:np.ndarray):
    '''
    Draw points with given coordinates.
    
    Parameters:
    fig: go.Figure - the figure to draw the points on
    points: list[list] - list of points
    '''
    if len(points) > 0:
        fig.update_traces(x=points[:, 0], y=points[:, 1], selector=dict(name='points'), overwrite=False)

def draw_balls(centers:list[list], R:float):
    '''
    Draw balls with given centers and radius.
    
    Parameters:
    fig: go.Figure - the figure to draw the balls on
    centers: list[list] - list of centers of the balls
    R: float - radius of the balls
    '''
    x_sphere, y_sphere = [], []
    for x, y in centers:
        phi = np.linspace(0, 2 * np.pi, 30)
        theta = np.linspace(0, np.pi, 30)
        phi, theta = np.meshgrid(phi, theta)
        
        x_sphere = np.concatenate((x_sphere, (R * np.sin(theta) * np.cos(phi)).flatten() + x, [None]))
        y_sphere =np.concatenate((y_sphere, (R * np.sin(theta) * np.sin(phi)).flatten() + y, [None]))
    return x_sphere, y_sphere
    
    
def draw_one_simplices(nodes:np.ndarray, edges:np.ndarray[np.ndarray]):
    '''
    Draw one-simplices (edges) with given nodes and edges.
    
    Parameters:
    fig: go.Figure - the figure to draw the one-simplices on
    nodes: list[list] - list of nodes
    edges: list[set] - list of edges
    '''
    if len(edges)> 0:
        x=np.concatenate([[ nodes[edges[i,0]][0], nodes[edges[i,1]][0], None] for i in range(len(edges))])
        y=np.concatenate([[ nodes[edges[i,0]][1], nodes[edges[i,1]][1], None] for i in range(len(edges))])
        return x,y           
    else:
        return [],[]
        
def draw_two_simplices(nodes:np.ndarray, triangles:np.ndarray[np.ndarray]):
    '''
    Draw two-simplices (triangles) with given nodes and triangles.
    
    Parameters:
    fig: go.Figure - the figure to draw the two-simplices on
    nodes: list[list] - list of nodes
    triangles: list[set] - list of triangles
    '''
    #if len(triangles) > 0:
        # fig.update_traces(x=np.concatenate([[None, nodes[triangles[i,0]][0], nodes[triangles[i,1]][0], nodes[triangles[i,2]][0], nodes[triangles[i,0]][0], None] for i in range(len(triangles))]),
        #                 y=np.concatenate([[None, nodes[triangles[i,0]][1], nodes[triangles[i,1]][1], nodes[triangles[i,2]][1], nodes[triangles[i,0]][1], None] for i in range(len(triangles))]),
        #                 selector=dict(name="triangles"),
        #                 overwrite=False)
    shapes = []
    for triangle in triangles:
            coords = nodes[triangle]
            path = f'M {coords[0][0]} {coords[0][1]} L {coords[1][0]} {coords[1][1]} L {coords[2][0]} {coords[2][1]} z'
            shapes.append(go.layout.Shape(
                        type="path",
                        path=path,
                        fillcolor="rgba(255, 210, 73, 0.5)",
                        line=dict(width=0),
                        layer="between",
                        editable=True
                ))
    return shapes
        
def all_circles_intersect(points, R):
    center_of_mass = np.mean(points, axis=0)
    return all(np.linalg.norm(center_of_mass-p) < R for p in points)


def filter_combinations_by_circle_intersection(points, R, comb_size):
    """Get combinations of points forming simplices where the circles intersect."""
    filtered_combinations = []
    for comb in combinations(range(len(points)), comb_size):
        if all_circles_intersect([points[i] for i in comb], R):
            filtered_combinations.append(comb)
    return filtered_combinations




if __name__ == "__main__":
    app.run_server(debug=True)