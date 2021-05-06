import json
import os

import networkx as nx
from networkx.readwrite import json_graph
from tensorflow.keras.utils import model_to_dot


def save_graph_plot(model, filename):
    file_basename, file_extension = os.path.splitext(filename)
    file_basename = file_basename + "_" + model.name
    file_extension = file_extension.strip('.')
    
    if file_extension == '':
        file_extension = 'pdf'

    pydot = model_to_dot(model,
                         show_layer_names=True,
                         show_shapes=True)
    binary_content = pydot.create(prog='dot',
                                  format=file_extension)
    string_content = binary_content.decode('utf-8')

    with open(f"{file_basename}.{file_extension}", "w") as f:
        f.write(string_content)


def save_graph_json(model, filename):
    file_basename, file_extension = os.path.splitext(filename)
    file_basename = file_basename + "_" + model.name
    file_extension = 'json'

    model_graph = model_to_networkx_graph(model)

    with open(f"{file_basename}.{file_extension}", "w") as f:
        json.dump(json_graph.node_link_data(model_graph), f, indent=4)


def model_to_networkx_graph(model):
   
    
    relevant_nodes = []
    for v in model._nodes_by_depth.values():
        relevant_nodes += v

    nodes = []
    links = []

    for layer in model.layers:
        layer_id = str(id(layer))

        # Iterate over inbound nodes to extract links that end in current layer
#         for node in layer._inbound_nodes:
#             if node in relevant_nodes:
#                 for inbound_layer, _, _, _ in node.iterate_inbound():
#                     inbound_layer_id = str(id(inbound_layer))
#                     links.append({
#                         "id_from": inbound_layer_id,
#                         "id_to": layer_id,
#                     })

        nodes.append({
            "id": layer_id,
            "name": layer.name,
            "cls_name": layer.__class__.__name__,
            "input_shape": layer.input_shape,
            "output_shape": layer.output_shape,
            "num_parameters": layer.count_params()
        })

    # Create directed networkx-graph
    nx_graph = nx.DiGraph()

    # Add nodes to graph and append attributes
    nx_graph.add_nodes_from([node["id"] for node in nodes])
    nx.set_node_attributes(nx_graph, {node["id"]: node for node in nodes})

    # Add edges to graph
    nx_graph.add_edges_from([(link["id_from"], link["id_to"]) for link in links])

    return nx_graph
