import math
import pickle
from pathlib import Path
from typing import Dict
import logging

import numpy as np
import networkx as nx
from tqdm import tqdm

from src.constants import DIMENSIONS, MIN_WEIGHT
from src.graph.utils import graph_hash
from src.graph.generator import normalize_edge_weights, prune_weights


def weight_to_PPMI(graph: nx.DiGraph) -> nx.DiGraph:
    unfrozen_graph = graph.copy()

    pr: Dict[str, float] = {}
    n = len(unfrozen_graph.nodes())
    for node in unfrozen_graph.nodes():
        total_weight = unfrozen_graph.in_degree(node, weight='weight')
        pr[node] = total_weight / n

    for _, v, a in tqdm(unfrozen_graph.edges(data=True), total=len(unfrozen_graph.edges())):
        ppmi = max(0, math.log(a['weight'] / pr[v], 2))
        a['weight'] = ppmi

    return unfrozen_graph

def graph_svd_decomp(graph: nx.DiGraph, min_weight: float=None) -> tuple:
    min_weight = min_weight or MIN_WEIGHT
    G = prune_weights(graph, min_weight)
    G = weight_to_PPMI(G)
    G = normalize_edge_weights(G)
    P = nx.to_numpy_array(G)
    S1 = P + P.transpose()
    S2 = S1 + np.matmul(S1, S1.transpose())
    svd_decomp = np.linalg.svd(S2)

    U, S, Vh = svd_decomp

    return G, U, S, Vh

def get_embeddings(graph: nx.DiGraph, dimensions: int = DIMENSIONS) -> Dict[str, np.ndarray]:
    _, U, S, _ = graph_svd_decomp(graph)
    embedding_fulldim = np.multiply(U, S)
    embedding = {k: v[:dimensions] for k, v in zip(list(graph.nodes()), embedding_fulldim)}
    return embedding

def cache_embeddings(graph: nx.DiGraph, dimensions: int = DIMENSIONS, cache_dir: Path = Path("cache")) -> Dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    graph_id = graph_hash(graph)
    cache_file = cache_dir / f"embeddings_{graph_id}_{dimensions}.pkl"

    if cache_file.exists():
        logging.info(f"Loading embeddings from cache: {cache_file}")
        with cache_file.open("rb") as f:
            return pickle.load(f)

    logging.info("Generating embeddings...")
    embeddings = get_embeddings(graph, dimensions)

    with cache_file.open("wb") as f:
        pickle.dump(embeddings, f)
        logging.info(f"Embeddings cached at: {cache_file}")

    return embeddings
