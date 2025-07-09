# `LAMBq` a lexical ambiguity quantification framework
<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none;"><img src="./repo_img.webp" alt="lamb_measures_dict" width="1000"></td>
    <td style="border: none;">
      <p>Lexical ambiguity is present in any word having more than one meaning. According to <a href="https://journals.sagepub.com/doi/abs/10.1177/1745691619885860">Rodd (2020)</a>, more than 80% of English words exhibit some degree of ambiguity. Despite the prevalence of the phenomenon, there is currently no method to quantify how ambiguous a word is.

Our work aims at bridging that gap by leveraging open, large-scale free association data and the computational linguistics toolkit. A detailed explanation of the methodology and evaluation results can be found in our accompanying paper, which will be available soon.</p>
    </td>
  </tr>
</table>



## Usage

### Setup and Installation

To run this project locally, follow these steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/iglesiasignacio/LAMBq 
   cd LAMBq
   ```

2. **Create and activate a virtual environment**  

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**  
    Install the necessary Python libraries via `requirements.txt`:  

    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare your data**  
    Ensure that the SWOW dataset file is available in the `data/` folder. Check the [SWOW research website](https://smallworldofwords.org/en/project/research) to download the required file. You will also need to update the `SWOW_PATH` variable in `src/constants.py` to point to the correct file path. The path points is pre-populated with `./data/SWOW-EN.R100.20180827.csv`, the file used for testing.
   

### Running the script  

To run the `main.py` script and calculate lexical ambiguity for input words, execute the following command:  

```bash  
python main.py --words <word1> <word2> ... [--postprocess]  
```  

#### Arguments:  
- words (_required_). A list of one or more words you want to analyze for lexical ambiguity.  
- postprocess (_optional_). If provided, this flag will apply additional post-processing to the detected communities to refine the results.

#### Output
The script will generate a report summarizing the lexical ambiguity of the input words. The report will be printed to the console. For example:  
```
Report for word: bank
===================

Meanings detected:
  1. account, card, count, deposit, institution, money, robber, safe, save, teller, vault (86.27%)
  2. river (13.73%)

Metrics:
  Normalized entropy: 0.577
  Normalized dissimilarity: 0.859
  Ambiguity: 0.495

----------------------------------------
```

## Citation
If you use this project in your work, please cite the following:  

> Iglesias I., Armstrong B., Laurino J., Kaczer L., Cabana A. *Automatic quantification of lexical ambiguity using large-scale word association data*. ~~Journal Name, Year. DOI or URL.~~
