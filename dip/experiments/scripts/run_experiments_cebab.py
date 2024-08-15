import json
import os
from argparse import ArgumentParser

import pandas as pd
from master_thesis.core.methods.probing_with_interventions import \
    ProbingWithInterventions
from master_thesis.core.methods.tcav import TCAV
from master_thesis.core.models.llama import load_model_and_tokenizer
from master_thesis.core.probes import load_cav
from master_thesis.core.utils.prompts import load_prompt
from master_thesis.core.utils.reproducibility import (save_results,
                                                      seed_everything)
from tqdm import tqdm


def load_experiment_config(config):
    with open(config, "r") as f:
        config = json.load(f)
    return config


def run_probing_with_intervention(
    run,
    config,
    dataset,
    model,
    tokenizer,
    tokens,
    label_prompt,
    aspect_prompt,
    save_path,
):
    cav = load_cav(
        f"{config['cavs_dir']}/{run['intervention']['model']}",
        f"{run['intervention']['aspect']}_{run['intervention']['layer']}",
    )

    probing_with_interventions = ProbingWithInterventions(
        cav=cav,
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        aspect_prompt=aspect_prompt,
        prompt=label_prompt,
        layer=run["intervention"]["layer"],
        device=config["device"],
        verbose=config["verbose"],
    )

    results = probing_with_interventions.evaluate(
        dataset,
        aspect_label_name=run["intervention"]["aspect"],
        original_label_name=config["original_aspect"],
    )
    save_results(
        results,
        class_names=config["class_names"],
        aspect_names=run["intervention"]["aspect_names"],
        save_path=save_path,
    )


def run_tcav(
    run,
    config,
    dataset,
    model,
    tokenizer,
    tokens,
    label_prompt,
    aspect_prompt,
    save_path,
):
    cav = load_cav(
        f"{config['cavs_dir']}/{run['intervention']['model']}",
        f"{run['intervention']['aspect']}_{run['intervention']['layer']}",
    )

    tcav = TCAV(
        cav=cav,
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        prompt=label_prompt,
        layer=run["intervention"]["layer"],
        device=config["device"],
        verbose=config["verbose"],
    )

    results = tcav.evaluate(dataset, aspect_label_name=run["intervention"]["aspect"])
    save_results(
        results,
        class_names=config["class_names"],
        aspect_names=run["intervention"]["aspect_names"],
        save_path=save_path,
    )


def run_experiment(run, config, model, tokenizer, label_prompt, aspect_prompt, tokens):
    save_path = f"{config['cache_dir']}/{run['method']}/{run['dataset']}/{run['intervention']['aspect']}_aspect_{run['intervention']['model']}_{run['intervention']['layer']}.json"
    if os.path.exists(save_path):
        print(f"Experiment already run.")
        return

    dataset = pd.read_csv(f"{config['dataset_dir']}/{run['dataset']}.csv")
    dataset = dataset[dataset[run["intervention"]["aspect"]] != -1]

    if run["method"] == "probing_with_intervention":
        run_probing_with_intervention(
            run,
            config,
            dataset,
            model,
            tokenizer,
            tokens,
            label_prompt,
            aspect_prompt,
            save_path,
        )

    if run["method"] == "tcav":
        run_tcav(
            run,
            config,
            dataset,
            model,
            tokenizer,
            tokens,
            label_prompt,
            aspect_prompt,
            save_path,
        )


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args = args.parse_args()

    config = load_experiment_config(args.config)

    model, tokenizer = load_model_and_tokenizer(config["model"])
    tokens = [
        tokenizer.encode(config["negative_token"])[-1],
        tokenizer.encode(config["positive_token"])[-1],
    ]

    original_prompt = load_prompt(
        config["data_dir"],
        dataset_path=config["prompt_dir"],
        prompt_type=config["prompt_type"],
        prompt_aspect=f"{config['original_aspect']}_aspect",
    )

    for run in tqdm(config["runs"]):
        print(
            f"Running {run['method']} on {config['dataset_dir']}/{run['dataset']}.csv for {run['intervention']['aspect']} aspect with {run['intervention']['model']} model and layer {run['intervention']['layer']}"
        )
        seed_everything()

        aspect_prompt = load_prompt(
            config["data_dir"],
            dataset_path=config["prompt_dir"],
            prompt_type=config["prompt_type"],
            prompt_aspect=f"{run['intervention']['aspect']}_aspect",
        )

        run_experiment(
            run, config, model, tokenizer, original_prompt, aspect_prompt, tokens
        )
