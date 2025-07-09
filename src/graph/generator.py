import pandas as pd
import networkx as nx
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict
from tqdm import tqdm

from src.constants import CUE_COL, RESP_COL, MIN_RESPONSES, MIN_WEIGHT
from src.graph.utils import graph_hash


def _normalize_graph_weights(G: nx.Graph) -> nx.Graph:
    out_degrees = {n: sum(attr['weight'] for _, _, attr in G.out_edges(n, data=True)) for n in G.nodes()}
    for u, v, attr in G.edges(data=True):
        weight = attr['weight']
        normalized_weight = weight / out_degrees[u]
        G[u][v]['weight'] = normalized_weight
    return G


def _normalize_dataframe_weights(edges_data: pd.DataFrame) -> pd.DataFrame:
    cue_weights = edges_data.groupby(CUE_COL)['weight'].sum().rename('sum_weights').reset_index()
    edges_data = edges_data.merge(cue_weights, on=CUE_COL)
    edges_data['weight'] = edges_data['weight'] / edges_data['sum_weights']
    return edges_data.drop(columns=['sum_weights'])


def normalize_edge_weights(
    data: Union[nx.Graph, pd.DataFrame]) -> Union[nx.Graph, pd.DataFrame]:
    if isinstance(data, nx.Graph):
        return _normalize_graph_weights(data)
    elif isinstance(data, pd.DataFrame):
        return _normalize_dataframe_weights(data)
    else:
        raise TypeError("Input must be either a networkx.Graph or a pandas.DataFrame.")

def prune_weights(G: nx.DiGraph, min_weight: float) -> nx.DiGraph:
    logging.info(f"Pruning edges with weight less than {min_weight}.")
    unfrozen_graph = G.copy()
    for u, v, attr in G.edges(data=True):
        if attr['weight'] < min_weight:
            unfrozen_graph.remove_edge(u, v)
    return unfrozen_graph

def preprocess_graph(
    word: str,
    data: pd.DataFrame,
    keep_edges: Optional[List[Tuple[str,str,Dict[str,float]]]] = None, 
    k: Optional[int] = None,
    responses: Optional[List[str]] = None,
    min_weight: Optional[float] = None
) -> nx.DiGraph:
    logging.info(f"Preprocessing graph for word: {word}, k={k}.")
    responses = responses or ['R1', 'R2', 'R3']
    min_weight = min_weight or MIN_WEIGHT
    if keep_edges is None:
        _, keep_edges = swow2graph(data)

    edges_data = swow2edges(word=word, data=data, keep_edges=keep_edges, k=k, responses=responses, normalize=True)
    subgraph = nx.DiGraph()
    subgraph.add_weighted_edges_from(list(edges_data.itertuples(index=False, name=None)))

    if min_weight and min_weight > 0.0:
        logging.info(f"Pruning edges with weight less than {min_weight}.")
        subgraph = subgraph.edge_subgraph(
            [(u, v) for (u, v, d) in subgraph.edges(data=True) if d['weight'] > min_weight]
        )

    return subgraph


def swow2graph(data: pd.DataFrame, cache_dir: Path = Path("cache")) -> nx.DiGraph:
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    graph_id = graph_hash(data)
    graph_cache_file = cache_dir / f"graph_{graph_id}.pkl"
    scc_cache_file = cache_dir / f"scc_edges_{graph_id}.pkl" 

    if graph_cache_file.exists():
        logging.info(f"Loading graph from cache: {graph_cache_file}")
        with graph_cache_file.open("rb") as f:
            graph = pickle.load(f)
    else:
        logging.info("Graph not cached. Generating edges data...")
        edges_data = swow2edges(word='*', data=data, responses=['R1', 'R2', 'R3'])
        filtered_edges = edges_data[edges_data[RESP_COL].isin(data[CUE_COL])][[CUE_COL, RESP_COL, 'weight']]
        edge_list = list(filtered_edges.itertuples(index=False, name=None))

        logging.info("Generating graph...")
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(edge_list)

        with graph_cache_file.open("wb") as f:
            pickle.dump(graph, f)
            logging.info(f"Graph cached at: {graph_cache_file}")

    largest_scc = max(nx.strongly_connected_components(graph), key=len, default=set())
    nodes_to_remove = set(graph.nodes) - largest_scc
    graph.remove_nodes_from(nodes_to_remove)

    scc_edges = [(u, v) for u, v in graph.edges(data=False) if u in largest_scc and v in largest_scc]

    if not scc_cache_file.exists():
        with scc_cache_file.open("wb") as f:
            pickle.dump(scc_edges, f)
            logging.info(f"SCC edges cached at: {scc_cache_file}")

    return graph, scc_edges


def swow2edges(
        word: str,
        data: pd.DataFrame,
        keep_edges: Optional[List[Tuple[str,str,Dict[str,float]]]] = None,
        k: int = 1,
        responses: Optional[List[str]] = None,
        normalize: bool = True,
        min_responses: int = None
    ) -> pd.DataFrame:

    min_responses = min_responses or MIN_RESPONSES
    responses = responses or ['R1']

    related_cues = {word}
    for _ in range(k - 1):
        next_cues = set(data.loc[data[CUE_COL].isin(related_cues), responses].values.flatten())
        new_cues = next_cues - related_cues
        if not new_cues:
            break
        related_cues.update(new_cues)
    
    filtered_data = data.loc[data[CUE_COL].isin(related_cues)]
    edges_data = pd.melt(
        filtered_data,
        id_vars=[CUE_COL],
        value_vars=responses,
        var_name='x',
        value_name=RESP_COL
    ).drop(columns=['x'])


    if keep_edges:
        edges_data['cue_resp_tuple'] = list(zip(edges_data[CUE_COL], edges_data[RESP_COL]))
        edges_data = edges_data[edges_data['cue_resp_tuple'].isin(keep_edges)]
        edges_data.drop(columns=['cue_resp_tuple'], inplace=True)

    edges_data = edges_data.groupby([CUE_COL, RESP_COL]).size().rename('weight').reset_index()
    edges_data = edges_data[edges_data['weight'] > min_responses].reset_index(drop=True)

    if normalize:
        edges_data = normalize_edge_weights(edges_data)

    if word == '*':
        logging.info("Processing all words.")
        all_edges = []
        for w in tqdm(data[CUE_COL].unique(), desc="Processing words"):
            edges_for_word = swow2edges(
                word=w,
                data=data,
                keep_edges=keep_edges,
                k=1,
                responses=responses,
                normalize=normalize,
                min_responses=min_responses
            )
            all_edges.append(edges_for_word)
        return pd.concat(all_edges, ignore_index=True)

    return edges_data