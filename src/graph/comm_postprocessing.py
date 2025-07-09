import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests

from src.constants import SWOW_PATH, SIMILARITY_THRESHOLD, DOMINANCE_THRESHOLD, ALPHA 
from src.data.loader import load_data
from src.graph.generator import swow2graph, swow2edges
from src.graph.node_embedding import cache_embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_similarity_matrix(words: List[str], embeddings: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    if embeddings is None:
        logging.info("Embeddings not provided. Loading data and generating embeddings.")
        data = load_data(SWOW_PATH)
        graph, _ = swow2graph(data)
        embeddings = cache_embeddings(graph)

    similarity_matrix = np.zeros((len(words), len(words)))
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            similarity_matrix[i][j] = cosine_similarity(embeddings[word1], embeddings[word2])
    return similarity_matrix

def outlier_words(words: List[str], similarity_matrix: np.ndarray, alpha: float = ALPHA,
                  correction: str = None,
                  test: str = "mannwhitney"):
  num_words = similarity_matrix.shape[0]

  if num_words < 3:
      return []

  mask = np.tril(np.ones_like(similarity_matrix), k=-1)
  all_similarities = similarity_matrix[mask == 1]
  baseline_mean = np.mean(all_similarities)

  p_values = np.zeros(num_words)

  if test == "z":
    leave_one_out_means = []
    for i in range(num_words):
      sm = np.delete(similarity_matrix, i, axis=0)
      sm = np.delete(sm, i, axis=1)
      mask = np.tril(np.ones_like(sm), k=-1)
      similarities = sm[mask == 1]
      leave_one_out_means.append(np.mean(similarities))

    leave_one_out_means = np.array(leave_one_out_means)
    std_dev = np.std(leave_one_out_means, ddof=1)
    if std_dev == 0:
      return []

    z_scores = (leave_one_out_means - baseline_mean) / std_dev
    p_values = 1 - norm.cdf(z_scores)

  elif test == "mannwhitney":
    for i in range(num_words):
      sm = np.delete(similarity_matrix, i, axis=0)
      sm = np.delete(sm, i, axis=1)
      mask = np.tril(np.ones_like(sm), k=-1)
      similarities_without = sm[mask == 1]

      _, p_value = mannwhitneyu(all_similarities,
                                similarities_without,
                                alternative='less',
                                method='auto')
      p_values[i] = p_value

  else:
    raise ValueError("unsupported test method")

  if correction == 'bonferroni':
    adjusted_alpha = alpha / num_words
    reject_flags = p_values < adjusted_alpha
  elif correction == 'fdr':
    reject_flags, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
  else:
    reject_flags = p_values < alpha

  outliers = []
  for word, is_rejected in zip(words, reject_flags):
    if is_rejected:
      print(word,' is an outlier in: ', words)
      outliers.append(word)

  return outliers

def improve_cd_for_word(
    word: str,
    data: pd.DataFrame,
    comm_detect_results_for_word,
    keep_edges,
    embeddings,
    similarity_threshold = SIMILARITY_THRESHOLD,
    dominance_threshold = DOMINANCE_THRESHOLD
    ):

  remove_word_flag = False

  G = comm_detect_results_for_word['graph']
  coms = comm_detect_results_for_word['comms']
  first_neighbours = [v for v in G[word]]

  coms_ = {}
  for key, value in coms.to_node_community_map().items():
    if key in first_neighbours:
      value_ = str(value[0])
      if value_ not in coms_:
        coms_[value_] = []
      coms_[value_].append(key)

  coms_dict = {w:k for k,v in coms_.items() for w in v}
  df_ = swow2edges(word=word, data=data, keep_edges=keep_edges, k=1, responses=['R1','R2','R3'], normalize=False)
  df_ = df_[df_.resp.isin(first_neighbours)]
  df_['community_id'] = df_['resp'].map(coms_dict) # df_ is a dataframe with 4 columns: [cue, resp, weight, community_id]

  similarities = {}
  for community, words in coms_.items():
    similarities[community] = []
    similarity_matrix = cosine_similarity_matrix(words, embeddings=embeddings)
    outliers = outlier_words(words, similarity_matrix, test="mannwhitney", correction="fdr")
    coms_[community] = [w for w in words if w not in outliers]
    words = coms_[community]

    if len(words) > 1:
      for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
          if i != j and j > i:
            sim = cosine_similarity(embeddings[word1], embeddings[word2])
            similarities[community].append(sim.item())
    else:
      similarities[community].append(1.)

  avg_within_comm_similarities = {k:sum(v)/len(v) for k,v in similarities.items()}
  sum_dominances = df_['weight'].sum()
  for comm in avg_within_comm_similarities:
    if avg_within_comm_similarities[comm] < similarity_threshold:
      # the community is "noisy"
      comm_dominance = df_.groupby(by=['community_id'])['weight'].sum()[comm]
      if comm_dominance/sum_dominances < dominance_threshold:
        # the community is not relevant in terms of dominance ==> remove community
        print(f"removing community: {coms_[comm]}")
        del coms_[comm]
      else:
        # the community is relevant in terms of dominance ==> remove word
        print(f"ambiguity calculation is not reliable for word: {word}")
        remove_word_flag = True
        break

  if len(coms_) == 0:
    remove_word_flag = True

  return remove_word_flag, coms_