import pandas as pd
import logging
import hashlib
from typing import List, Union, Tuple, Any
import networkx as nx


def graph_hash(graph_input: Union[nx.Graph, pd.DataFrame]) -> str:
    if isinstance(graph_input, nx.Graph):
        edge_list = sorted(graph_input.edges(data=True))
        edge_list_str = str(edge_list)
        return hashlib.sha256(edge_list_str.encode()).hexdigest()
    elif isinstance(graph_input, pd.DataFrame):
        serialized = graph_input.sort_values(by=graph_input.columns.tolist())
        serialized_str = serialized.to_csv(index=False).encode('utf-8')
        return hashlib.sha256(serialized_str).hexdigest()
    else:
        raise ValueError("Input must be a NetworkX graph or a pandas DataFrame as required by `swow2graph`.")