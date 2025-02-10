import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from networkx.drawing.nx_agraph import graphviz_layout

class Draw:
    def __init__(self, node_types: dict, edge_types: dict) -> None:
        self.NODE_TYPES = node_types
        self.EDGE_TYPES = edge_types
        
    def dfg(self, G: nx.DiGraph, save_path: str=None):  
        file_name = os.path.basename(save_path) if save_path else "data_flow_graph"
        
        directory_name = os.path.dirname(save_path) if save_path else "output"
        directory = os.path.join(os.getcwd(), directory_name)
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)    

        plt.figure(figsize=(32, 64))
        nx.nx_agraph.write_dot(G, os.path.join(directory, f"{file_name}.dot"))
        for node in G.nodes:
            G.nodes[node]["width"] = 0.9
        pos = graphviz_layout(G, prog="dot")
          
        node_labels = {}
        for key, label in nx.get_node_attributes(G, "label").items():
            node_labels[key] = label
        
        light_steel_blue = "#B0C4DE"
        very_soft_blue = "#b3b0de"
        pink = "#FFC0CB"
        light_green = "#c4deb0"
        very_soft_yellow = "#DEDAB0"
        very_soft_lime_green = "#B0DEB9"
        very_soft_purple = "#DEB0DE"
        
        light_salmon = "#FFA07A"
        pale_green = "#98FB98"
        gray = "#d3d3d3"
        powder_blue = "#B0E0E6"
        peach_puff = "#FFDAB9"
        
        node_colors = []
        for node in G.nodes:
            if G.nodes[node]["node_type"] == self.NODE_TYPES["VARIABLE"]:
                node_colors.append(light_steel_blue)
            elif G.nodes[node]["node_type"] == self.NODE_TYPES["INTERMEDIATE"]:
                node_colors.append(very_soft_blue)
            elif G.nodes[node]["node_type"] == self.NODE_TYPES["FUNCTION"]:
                node_colors.append(light_green)
            elif G.nodes[node]["node_type"] == self.NODE_TYPES["IMPORT"]:
                node_colors.append(pink)
            elif G.nodes[node]["node_type"] == self.NODE_TYPES["IF"]:
                node_colors.append(very_soft_yellow)
            elif G.nodes[node]["node_type"] == self.NODE_TYPES["LOOP"]:
                node_colors.append(very_soft_lime_green)
            elif G.nodes[node]["node_type"] == self.NODE_TYPES["CLASS"]:
                node_colors.append(very_soft_purple)
            else:
                node_colors.append("black")
        
        edge_labels = {}
        for key, label in nx.get_edge_attributes(G, "code").items():
            edge_labels[key] = label
        for key, count in nx.get_edge_attributes(G, "count").items():
            if count > 1:
                edge_labels[key] += f" ({count}x)"
        
        edge_colors = []
        edge_styles = []
        for edge in G.edges:
            if G.edges[edge]["edge_type"] == self.EDGE_TYPES["CALLER"]:
                edge_colors.append(light_salmon)
                edge_styles.append("solid")  # Caller edges are solid
            elif G.edges[edge]["edge_type"] == self.EDGE_TYPES["OMITTED"]:
                edge_colors.append(gray)
                edge_styles.append("dashed")  # Omitted edges are dotted
            elif G.edges[edge]["edge_type"] == self.EDGE_TYPES["INPUT"]:
                edge_colors.append(pale_green)
                edge_styles.append("solid")
            elif G.edges[edge]["edge_type"] == self.EDGE_TYPES["BRANCH"]:
                edge_colors.append(powder_blue)
                edge_styles.append("solid")  # Caller edges are solid
            elif G.edges[edge]["edge_type"] == self.EDGE_TYPES["LOOP"]:
                edge_colors.append(peach_puff)
                edge_styles.append("solid")  # Caller edges are solid
            elif G.edges[edge]["edge_type"] == self.EDGE_TYPES["FUNCTION_CALL"]:
                edge_colors.append(light_green)
                edge_styles.append("solid")  # Caller edges are solid            
            else:
                edge_colors.append("blue")
                edge_styles.append("dotted")
        
        for i, edge in enumerate(G.edges):
            nx.draw_networkx_edges(
                G, 
                pos, 
                edgelist=[edge], 
                edge_color=[edge_colors[i]], 
                style=edge_styles[i], 
                width=2, 
                arrows=True, 
                arrowstyle="-|>", 
                arrowsize=20,
            )
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="#454545")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#454545")
        
        # Custom legend for node types
        node_legend_elements = [
            Patch(facecolor=light_steel_blue, edgecolor="none", label="Variable") if light_steel_blue in node_colors else None,
            Patch(facecolor=very_soft_blue, edgecolor="none", label="Intermediate") if very_soft_blue in node_colors else None,
            Patch(facecolor=light_green, edgecolor="none", label="Function") if light_green in node_colors else None,
            Patch(facecolor=pink, edgecolor="none", label="Import") if pink in node_colors else None,
            Patch(facecolor=very_soft_yellow, edgecolor="none", label="If") if very_soft_yellow in node_colors else None,
            Patch(facecolor=very_soft_lime_green, edgecolor="none", label="Loop") if very_soft_lime_green in node_colors else None,
            Patch(facecolor=very_soft_purple, edgecolor="none", label="Class") if very_soft_purple in node_colors else None,
            Patch(facecolor="black", edgecolor="none", label="Change") if "black" in node_colors else None
        ]
        node_legend_elements = [elem for elem in node_legend_elements if elem is not None]
        
        # Custom legend for edge types
        edge_legend_elements = [
            Line2D([0], [0], color=light_salmon, lw=2, label="Caller") if light_salmon in edge_colors else None,
            Line2D([0], [0], color=pale_green, lw=2, label="Input") if pale_green in edge_colors else None,
            Line2D([0], [0], color=gray, lw=2, linestyle="--", label="Omitted") if gray in edge_colors else None,
            Line2D([0], [0], color=powder_blue, lw=2, label="Branch") if gray in edge_colors else None,
            Line2D([0], [0], color=peach_puff, lw=2, label="Loop") if gray in edge_colors else None,
            Line2D([0], [0], color=light_green, lw=2, label="Function Call") if light_green in edge_colors else None,
            Line2D([0], [0], color="blue", lw=2, linestyle=":", label="Change") if "blue" in edge_colors else None
        ]
        edge_legend_elements = [elem for elem in edge_legend_elements if elem is not None]
        
        # Combine legends and display them
        plt.legend(handles=node_legend_elements + edge_legend_elements, loc="upper left", fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(directory, f"{file_name}.png"))
        plt.clf()
        plt.close()
            
        
    def labelled_dfg(self, G: nx.DiGraph, save_path: str = None):  
        file_name = os.path.basename(save_path) if save_path else "data_flow_graph"
        directory_name = os.path.dirname(save_path) if save_path else "output"
        directory = os.path.join(os.getcwd(), directory_name)
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)    
        
        plt.figure(figsize=(32, 64))
        nx.nx_agraph.write_dot(G, os.path.join(directory, f"{file_name}.dot"))
        
        pos = graphviz_layout(G, prog="dot")
        
        edge_labels = nx.get_edge_attributes(G, "code")
        
        # Define the domain label color map
        domain_color_map = {
            'MODEL_TRAIN': "#B0C4DE",
            'MODEL_EVALUATION': "#b3b0de",
            'HYPERPARAMETER_TUNING': "#FFC0CB",
            'DATA_EXPORT': "#c4deb0",
            'DATA_IMPORT_EXTRACTION': "#DEDAB0",
            'DATA_TRANSFORM': "#B0DEB9",
            'EDA': "#DEB0DE",
            'ENVIRONMENT': "#FFA07A",
            'NOT_INTERESTING': "#d3d3d3",
        }
        
        edge_domain_labels = nx.get_edge_attributes(G, "domain_label")
        node_labels = {node: str(node) for node in G.nodes} 
        
        edge_colors = [domain_color_map.get(edge_domain_labels.get(edge, "NOT_INTERESTING"), "black") for edge in G.edges]
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrows=True, arrowstyle="-|>", arrowsize=20)
        nx.draw_networkx_nodes(G, pos, node_color="lightgrey", node_size=1000)  # Nodes are now a neutral color
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#454545")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="black")  # Display node IDs
        
        edge_legend_elements = [Line2D([0], [0], color=color, lw=2, label=label) for label, color in domain_color_map.items()]
        
        plt.legend(handles=edge_legend_elements, loc="upper left", fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(directory, f"{file_name}.png"))
        plt.clf()
        plt.close()


    def ast(self, G: nx.DiGraph, t2t_paths: list[list[int]]):
        if not os.path.exists("output"):
            os.makedirs("output", exist_ok=True)

        G = G.copy()
        plt.figure(figsize=(128, 32))
        nx.nx_agraph.write_dot(G, os.path.join("output", "ast.dot"))
        for node in G.nodes:
            G.nodes[node]["width"] = 0.9
        pos = graphviz_layout(G, prog="dot")
        labels = nx.get_node_attributes(G, "label")
        code = nx.get_node_attributes(G, "code")

        node_labels = {}
        for key, label in labels.items():
            node_labels[key] = f"{label}\n{code[key]}"
        
        edge_colors = []
        edge_to_index = {}
        for index, edge in enumerate(G.edges):
            edge_to_index[tuple(edge)] = index
            edge_colors.append("black")
        
        for edge in t2t_paths:
            color = np.random.choice(["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#bb3e03", "#ae2012", "#9b2226"])
            for node1 in edge:
                for node2 in edge:
                    if node1 != node2:
                        if (node1, node2) in edge_to_index:
                            edge_colors[edge_to_index[(node1, node2)]] = color
        
        node_colors = ["#eeeeee" for _ in range(len(G.nodes))]

        nx.draw(G, pos, labels=node_labels, with_labels=True, edge_color=edge_colors, font_size=10, font_color="#454545", node_color=node_colors, node_size=2500)
        plt.savefig(os.path.join("output", f"ast_graph.png"))
        plt.clf()