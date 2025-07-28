import argparse
from src.pipeline import run_ambiguity_pipeline
from src.lexical_ambiguity import generate_report

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate lexical ambiguity for input words.")
    parser.add_argument("--words", nargs="+", required=True, help="Word(s) to analyze.")
    parser.add_argument("--postprocess", action="store_true", default=False,
                        help="Add flag for communities to be post-processed.")
    return parser.parse_args()

def main():
    args = parse_args()
    amb_results = run_ambiguity_pipeline(args.words, postprocess=args.postprocess)
    generate_report(amb_results)

if __name__ == "__main__":
    main()