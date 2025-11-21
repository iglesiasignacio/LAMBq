import logging
from src.data.loader import load_data
from src.constants import SWOW_PATH, DIMENSIONS
from src.lexical_ambiguity import comms_to_amb
from src.graph.generator import swow2graph
from src.graph.node_embedding import cache_embeddings
from src.graph.comm_detection import comm_detect
import warnings

warnings.filterwarnings("ignore", message=".*(bayanpy|graph_tool|pyclustering|ASLPAw).*")
logging.basicConfig(format='%(module)s - %(message)s', level=logging.INFO)

def run_ambiguity_pipeline(words, postprocess=False, **kwargs):
    # load | generate
    data = load_data(SWOW_PATH)
    graph, scc_edges = swow2graph(data)
    embeddings = cache_embeddings(graph, dimensions=DIMENSIONS)
    # detect communities
    comms = comm_detect(data=data, words=words, keep_edges=scc_edges)
    # ambiguity calculation
    amb_results = comms_to_amb(
        data=data,
        comm_detect_results=comms,
        keep_edges=scc_edges,
        embeddings=embeddings,
        improve=postprocess,
        **kwargs
    )
    return amb_results