"""Estimation of CONEXP (Conditional Expectation)"""

import numpy as np
import torch as t
from tqdm import tqdm


class CONEXPEstimator:
    """
    Estimation of CONEXP (Conditional Expectation).

    Sources:
    * Explaining Classifiers with Causal Concept Effect (CaCE)
    * CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior

    The effect of a concept C is the average difference in predictions on examples with different values of C

    CONEXP^D_N_\phi (C, c, c') = 1 \ |D^{C=c'}|  sum(N(\phi(x_c'))) - 1 \ |D^{C=c}|  sum(N(\phi(x_c)))

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

    def evaluate(self, dataset, aspect_label):
        results = {0: [], 1: []}

        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            aspect = row[aspect_label]
            if aspect == -1:
                continue
            sentence = row["sentence"]
            results[aspect].append(self.get_token_probability(sentence))

        negative_aspect_mean = np.mean(results[0], axis=0)
        positive_aspect_mean = np.mean(results[1], axis=0)

        negative_to_positive = abs(positive_aspect_mean - negative_aspect_mean)
        positive_to_negative = abs(negative_aspect_mean - positive_aspect_mean)

        results = {
            0: {i: score for i, score in enumerate(negative_to_positive)},
            1: {i: score for i, score in enumerate(positive_to_negative)},
        }

        return results

    def get_token_probability(self, sentence: str) -> float:
        input_text = self.prompt.format(sentence=sentence)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )
        probs = self.model(input_ids).logits[0, -1, :].softmax(-1)
        return np.array([probs[token].item() for token in self.tokens])
