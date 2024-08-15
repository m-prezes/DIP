"""Estimation of groud truth CaCE (Causal Concept Effect)"""

import numpy as np
import torch as t
from tqdm import tqdm


class GTCaCEEstimator:
    """
    Estimation of groud truth CaCE (Causal Concept Effect)

    Sources:
    * Explaining Classifiers with Causal Concept Effect (CaCE)
    * CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior

    Definition 1 (Empirical Individual Causal Concept Effect; ICaCE).
    For a neural network N and feature function \phi, the empirical individual causal concept effect
    of changing the value of concept C from c to c' for state of affairs u is:

    ICaCE_N_\phi (x^{C=c}_u, x^{C=c'}_u) = N(\phi(x^{C=c}_u)) - N(\phi(x^{C=c'}_u))

    where (x^{C=c}_u, x^{C=c'}_u) is a tuple of inputs originating from u
    with the concept C set to the values c and c', respectively.


    Definition 2 (Empirical Causal Concept Effect; CaCE).
    For a neural network N and feature function \phi, the empirical causal concept effect
    of changing the value of concept C from c to c' in dataset D is:

    CaCE^D_N_\phi (C, c, c') = 1 \ |D^{c->c'}_C|  sum( ICaCE_N_\phi (x^{C=c}_u, x^{C=c'}_u) )

    """

    def __init__(
        self,
        model,
        tokenizer,
        prompt,
        tokens,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt = prompt
        self.tokens = tokens
        self.verbose = verbose

    def evaluate(self, dataset, root_dataset, aspect_label, original_label="label"):
        results = {0: [], 1: []}

        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            aspect = row[aspect_label]
            if aspect == -1:
                continue
            sentence, contrfact = self.prepare_pair(
                row, root_dataset, aspect_label, original_label
            )
            if contrfact is None:
                continue
            results[aspect].append(self.get_icace(sentence, contrfact))

            if self.verbose:
                print(f"Aspect: {aspect} | ICACE: {results[aspect][-1]}")
                print(f"{sentence} | {contrfact}")

        return self.postprocess_results(results)

    def prepare_pair(self, row, root_dataset, aspect_label, original_label):
        sentence = row["sentence"]
        sentences_set_id = row["set_id"]

        sentences_set = root_dataset[root_dataset["set_id"] == sentences_set_id]
        sentences_set = sentences_set[sentences_set["sentence"] != sentence]
        sentences_set = sentences_set[sentences_set[aspect_label] != -1]

        # get columns from row
        const_columns = root_dataset.columns.difference(
            ["id", "sentence", "set_id", aspect_label, original_label]
        )

        contrfact = sentences_set[
            (sentences_set[aspect_label] != row[aspect_label])
            & (sentences_set[const_columns].eq(row[const_columns]).all(axis=1))
        ]

        if len(contrfact) == 0:
            return sentence, None

        contrfact = contrfact.sample(1).iloc[0]["sentence"]
        return sentence, contrfact

    def get_token_probability(self, sentence: str) -> float:
        input_text = self.prompt.format(sentence=sentence)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )
        probs = self.model(input_ids).logits[0, -1, :].softmax(-1)
        return np.array([probs[token].item() for token in self.tokens])

    def get_icace(self, sentence, contrfact):
        sentence_proba = self.get_token_probability(sentence)
        contrfact_proba = self.get_token_probability(contrfact)
        return abs(sentence_proba - contrfact_proba)

    def postprocess_results(self, results):
        negative_to_positive = np.array(results[0] if len(results[0]) > 0 else [0])
        positive_to_negative = np.array(results[1] if len(results[1]) > 0 else [0])

        results = {
            0: {
                0: np.mean(negative_to_positive[:, 0]),
                1: np.mean(negative_to_positive[:, 1]),
            },
            1: {
                0: np.mean(positive_to_negative[:, 0]),
                1: np.mean(positive_to_negative[:, 1]),
            },
        }

        return results
