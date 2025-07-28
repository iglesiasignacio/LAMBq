import pandas as pd
import logging
from typing import List, Optional, Tuple, Dict
from cdlib import algorithms

from src.constants import MIN_RESPONSES, MIN_WEIGHT
from src.graph.generator import preprocess_graph, swow2graph


def comm_detect(
    words: List[str],
    data: pd.DataFrame,
    keep_edges: Optional[List[Tuple[str,str,Dict[str,float]]]] = None,
    responses: List[str] = None,
    min_weight: float=None,
    algorithm=algorithms.infomap,
    **kwargs
):
    responses = responses or ['R1', 'R2', 'R3']
    min_weight = min_weight or MIN_WEIGHT
    if keep_edges is None:
        _, keep_edges = swow2graph(data)

    results = {}

    for idx, word in enumerate(words):
        logging.info(f"Processing word {idx + 1}/{len(words)}: {word}")
        graph = preprocess_graph(
            data=data,
            word=word,
            keep_edges=keep_edges,
            k=2,
            responses=responses,
            min_weight=min_weight
        )
        if len(graph.nodes()) > 0:
            results[word] = {'graph': graph}
            communities = algorithm(graph, **kwargs)
            results[word]['comms'] = communities
        else:
            raise ValueError(f'{word} not in graph')

    return results
