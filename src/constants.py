# data
SWOW_PATH = './data/SWOW-EN.R100.20180827.csv' # path to dataset containing SWOW data
CUE_COL = 'cue'
RESP_COL = 'resp'

# embeddings
DIMENSIONS = 400 # embedding dimensions

# graph processing
MIN_RESPONSES = 2 # minimum number of responses for a cue-response edge to be considered
MIN_WEIGHT = 0.02 # optimized pruning weight

# postprocessing
SIMILARITY_THRESHOLD = 0.3 
DOMINANCE_THRESHOLD = 0.3 
ALPHA = 0.05