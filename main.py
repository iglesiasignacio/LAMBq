import argparse
import warnings
import logging

from src.data.loader import load_data
from src.constants import SWOW_PATH, DIMENSIONS
from src.lexical_ambiguity import comms_to_amb, generate_report
from src.graph.generator import swow2graph
from src.graph.node_embedding import cache_embeddings
from src.graph.comm_detection import comm_detect

warnings.filterwarnings("ignore", message=".*(bayanpy|graph_tool|pyclustering|ASLPAw).*")
logging.basicConfig(format='%(module)s - %(message)s', level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate lexical ambiguity for input words.")
    parser.add_argument("--words", nargs="+", required=True, help="Word(s) to analyze.")
    parser.add_argument("--postprocess", action="store_true", default=False, help="Add flag for communities to be post-processed.")
    return parser.parse_args()

def main():
    args = parse_args()
    # load | generate
    data = load_data(SWOW_PATH)
    graph, scc_edges = swow2graph(data)
    embeddings = cache_embeddings(graph, dimensions=DIMENSIONS)
    # detect communities
    comms = comm_detect(data=data, words=args.words, keep_edges=scc_edges)
    # ambiguity calculation
    amb_results = comms_to_amb(data=data, comm_detect_results=comms, keep_edges=scc_edges, embeddings=embeddings, improve=args.postprocess)
    generate_report(amb_results)
    
if __name__ == "__main__":
    main()
