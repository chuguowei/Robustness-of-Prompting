import warnings
import json
import os
import random
import argparse
import torch
import pandas as pd

from config import CHOICE_DATASETS, QUESTION_DATASETS, YESNO_DATASETS
from data_utils import preprocess_datasets_inplace
from model_utils import create_bnb_config, load_model
from steering import SteeringManager
from scoring import ConsistencyScorer
from inference import run_experiment

warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

ALL_DATASETS = ['aqua', 'GSM8K', 'addsub', 'CSQA', 'bigbench_date',
                'object_tracking', 'singleeq', 'MultiArith', 'SVAMP', 'Strategy']


def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive Activation Steering")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        metavar='DATASET',
        help=f'Datasets to run. Choices: {ALL_DATASETS}. Default: all.'
    )
    parser.add_argument(
        '--robust_types',
        type=str,
        default='r1',
        choices=['r1','r2','r3','r4', 'r5'],
        help='Perturbation type. Default: r1.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='../Meta-Llama-3-8B-Instruct',
        help='Path or name of the model. Default: ../Meta-Llama-3-8B-Instruct.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _mn = args.model_name
    if os.path.isabs(_mn) or _mn.startswith('../') or _mn.startswith('./'):
        model_name = _mn
    else:
        model_name = f'../{_mn}'
    datasets     = args.datasets
    robust_types = args.robust_types

    preprocess_datasets_inplace(datasets, robust_types)

    print(f"Loading model: {model_name}")
    bnb_config        = create_bnb_config()
    model, tokenizer  = load_model(model_name, bnb_config)

    steering_manager   = SteeringManager(model, tokenizer)
    consistency_scorer = ConsistencyScorer(model, tokenizer, steering_manager)

    summary_file = (f'result/'
                    f'summary_results_consistency_clamp_{robust_types}.csv')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    if not os.path.exists(summary_file):
        pd.DataFrame(columns=['dataset', 'robust_type', 'best_layers', 'accuracy']) \
          .to_csv(summary_file, index=False)

    for dataset in datasets:
        data_path = f"data/data_{robust_types}/rewritten_{dataset}_1.json"
        if not os.path.exists(data_path):
            print(f"Skipping {dataset}: file not found at {data_path}")
            continue

        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        total_data_len = len(full_data)
        print(f"[{dataset}] Shuffling & splitting (Dev=1/4, Eval=3/4)...")
        random.shuffle(full_data)

        dev_set_size = max(1, int(total_data_len / 4))
        dev_set      = full_data[:dev_set_size]
        eval_set     = full_data[dev_set_size:]
        print(f"[{dataset}] Total: {total_data_len} | Dev: {len(dev_set)} | Eval: {len(eval_set)}")

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            all_layers = list(range(len(model.model.layers)))
        else:
            all_layers = list(range(len(model.layers)))

        steering_manager.compute_all_vectors(dev_set, all_layers, dataset=dataset)

        consistency_scores = consistency_scorer.score(all_layers)

        torch.cuda.empty_cache()
        print(f"\n[*] Starting Formal Inference on {dataset} "
              f"(best_layers={steering_manager.best_layers})...")

        acc = run_experiment(
            dataset, robust_types,
            model, tokenizer, steering_manager,
            eval_data=eval_set
        )
        print(f"[{dataset}] Final Accuracy = {acc:.2%}")

        new_row = pd.DataFrame([{
            'dataset':     dataset,
            'robust_type': robust_types,
            'best_layers': str(steering_manager.best_layers),
            'accuracy':    acc
        }])
        if os.path.exists(summary_file):
            existing   = pd.read_csv(summary_file)
            updated_df = pd.concat([existing, new_row], ignore_index=True)
        else:
            updated_df = new_row
        updated_df.to_csv(summary_file, index=False)

        steering_manager.steering_vectors.clear()
        steering_manager.layer_norms.clear()
        steering_manager.layer_norm_upper.clear()
        steering_manager.layer_mu_clean.clear()
        steering_manager.layer_mu_perturbed.clear()
        steering_manager.raw_diffs.clear()
        steering_manager.dev_weights.clear()
        steering_manager.best_layers = []
        torch.cuda.empty_cache()

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()