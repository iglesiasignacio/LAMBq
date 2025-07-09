import logging
from tqdm import tqdm
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.graph.generator import swow2edges
from src.graph.comm_postprocessing import improve_cd_for_word


def entropy(p: List[float]) -> float:
    if not isinstance(p, list):
        raise TypeError(f'Input `p` is not of type list but {type(p)}')
    elif len(p) == 0:
        raise ValueError('Input `p` is empty')
    elif len([p_i for p_i in p if p_i < 0]) > 0:
        raise ValueError(f'At least one element in input `p` is lower than zero')

    return -sum([p_i * math.log2(p_i/sum(p)) for p_i in p if p_i > 0])/sum([p_i for p_i in p if p_i > 0])

def SVD_similarity(S: List[float], input_normalized: bool = True, max_norm: float = 1.) -> float:
    '''
    Takes an array of eigenvalues.
    
    eigenvalues come from SVD of a:
        a) column normalized matrix
        b) column raw matrix
    (each of these have different implications and require different normalization as well)
    
    Returns a normalized dissimilarity coefficient.
    '''
    if len(S) > 1:
        if input_normalized:
            return (S[0]**2 - 1)/(len(S) - 1)
        else:
            return S[0]/(len(S)*max_norm)
    else:
        return 1

def ambiguity(normalized_avg_coms_embeddings: np.ndarray, norms: List[float]) -> Tuple[float, float, float]:
    '''
    Computes the entropy and dissimilarity coefficients for the community embeddings.
    
    Returns:
        - norm_entropy: Entropy of norms
        - norm_dissimilarity: Dissimilarity coefficient based on SVD
        - amb: Combined measure of ambiguity (entropy * dissimilarity)
    '''
    if len(norms) > 1:
        _, S, _ = np.linalg.svd(normalized_avg_coms_embeddings, full_matrices=True)
        norm_dissimilarity = 1 - SVD_similarity(S, input_normalized=True)
        norm_entropy = entropy(norms)/math.log2(len(norms))
        return norm_entropy, norm_dissimilarity, norm_dissimilarity * norm_entropy
    else:
        return 0, 0, 0

def comms_to_amb(
        data: pd.DataFrame,
        comm_detect_results: Dict[str, Dict], 
        keep_edges: Optional[List[Tuple[str,str,Dict[str,float]]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        improve: bool = True) -> Dict[str, Dict]:
    '''
    Processes community detection results to calculate ambiguity for each word.
    
    Parameters:
        comm_detect_results: A dictionary where keys are words and values are dictionaries with graph and community info
        embeddings: A dictionary where keys are words and values are their embeddings
        improve: Flag to improve community detection using a post-processing method
    
    Returns:
        A dictionary with calculated ambiguity values for each word
    '''
    amb_results_dict = {}

    target_words = [w for w in comm_detect_results]  # the words for which communities were detected

    coms_inv_dict = {}
    for word in target_words:
        r = comm_detect_results[word]
        G = r['graph']
        first_neighbours = [v for v in G[word]]

        if not improve:
            remove_word_flag = False
            coms = r['comms']
            coms_ = {}
            for key, value in coms.to_node_community_map().items():
                if key in first_neighbours:
                    value_ = str(value[0])
                    if value_ not in coms_:
                        coms_[value_] = []
                    coms_[value_].append(key)

            coms_inv_dict[word] = coms_

        else:
            remove_word_flag, coms_ = improve_cd_for_word(word=word, data=data, keep_edges=keep_edges, comm_detect_results_for_word=r, embeddings=embeddings)

        if not remove_word_flag:
            coms_dict = {w: k for k, v in coms_.items() for w in v}
            df_ = swow2edges(word=word, data=data, keep_edges=keep_edges, k=1, responses=['R1', 'R2', 'R3'], normalize=False)
            df_ = df_[df_.resp.isin(first_neighbours)]
            df_['community_id'] = df_['resp'].map(coms_dict)  # df_ is a dataframe with 4 columns: [cue, resp, weight, community_id]

            norms = df_.groupby(by=['community_id'])['weight'].sum().to_list()
            coms_dict_inv = df_.groupby(by=['community_id'])['resp'].apply(list).to_dict()

            coms_avg_embeddings = {}
            com_idxs_ok = []
            for idx, com in enumerate(coms_dict_inv):
                listed_emb = [embeddings[word].reshape(1, -1) for word in coms_dict_inv[com] if word in embeddings]
                emb_concat = np.concatenate(listed_emb)
                coms_avg_embeddings[com] = np.mean(emb_concat, axis=0)
                com_idxs_ok.append(idx)

            avg_embeddings_stacked = np.concatenate([coms_avg_embeddings[com].reshape(1, -1) for com in coms_avg_embeddings])
            emb_norms = [np.linalg.norm(avg_embeddings_stacked[i, :]) for i in range(avg_embeddings_stacked.shape[0])]
            normalized_emb = [avg_embeddings_stacked[i, :] / emb_norms[i] for i in range(avg_embeddings_stacked.shape[0])]
            normalized_avg_coms_embeddings = np.array(normalized_emb)
            norms = [norms[i] for i in com_idxs_ok]  # dropping empty coms

            e, d, amb = ambiguity(normalized_avg_coms_embeddings, norms)

            amb_results_dict[word] = {
                'coms_assigned': coms_dict_inv,
                'normalized_avg_coms_embeddings': normalized_avg_coms_embeddings,
                'norms': norms,
                'entropy_coef': e,
                'disimilarity_coef': d,
                'ambiguity': amb
            }

        else:
            logging.info(f"Removed by post-process: {word}")

    return amb_results_dict

def generate_report(data):
    """
    Generates a formatted report from the input data.

    Args:
        data (dict): A dictionary containing items to report.

    Returns:
        None
    """
    for word, details in data.items():
        print(f"\nReport for word: {word}\n" + "=" * (15 + len(word)))

        # Extract meanings and normalize their frequencies
        meanings = details['coms_assigned']
        total_norm = sum(details['norms'])
        percentages = [(count / total_norm) * 100 for count in details['norms']]

        print("\nMeanings detected:")
        for i, (meaning, percentage) in enumerate(zip(meanings.values(), percentages), start=1):
            meaning_str = ', '.join(meaning)
            print(f"  {i}. {meaning_str} ({percentage:.2f}%)")

        # Extract and format coefficients
        normalized_entropy = round(details['entropy_coef'], 3)
        normalized_dissimilarity = round(details['disimilarity_coef'], 3)
        ambiguity = round(details['ambiguity'], 3)

        print("\nMetrics:")
        print(f"  Normalized entropy: {normalized_entropy}")
        print(f"  Normalized dissimilarity: {normalized_dissimilarity}")
        print(f"  Ambiguity: {ambiguity}")

        print("\n" + "-" * 40 + "\n")
