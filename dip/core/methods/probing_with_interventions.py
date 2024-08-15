import numpy as np
import torch
from tqdm import tqdm


class ProbingWithInterventions:
    def __init__(
        self,
        cav,
        model,
        tokenizer,
        tokens,
        prompt,
        aspect_prompt,
        layer,
        device,
        verbose=True,
    ):
        self.cav = cav
        self.model = model
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.device = device
        self.verbose = verbose
        self.prompt = prompt
        self.aspect_prompt = aspect_prompt
        self.layer = layer
        self.tot = 0.7
        self.tick = 0.13

    def evaluate(self, dataset, aspect_label_name, original_label_name):
        results = {0: {0: [], 1: []}, 1: {0: [], 1: []}}

        for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
            sentence = row["sentence"]
            label = row[original_label_name]
            aspect_label = row[aspect_label_name]

            aspect_intervention_direction = (
                "true2false" if aspect_label == 1 else "false2true"
            )
            intervention_cav = self.adapt_cav_intervention_vector(
                sentence, self.cav, aspect_intervention_direction
            )
            if intervention_cav is None:
                results[aspect_label][label].append(0)
                results[aspect_label][abs(label - 1)].append(0)
                print("Skipping")
                continue

            def hook_intervention(module, input, output):
                output[0][:, -1, :] += intervention_cav
                return output

            handle = self.model.model.layers[self.layer].register_forward_hook(
                hook_intervention
            )

            probs = self.get_probs(sentence, self.prompt)
            handle.remove()

            if sum(probs) < self.tot:
                results[aspect_label][label].append(0)
                results[aspect_label][abs(label - 1)].append(0)
            else:
                diff = probs[1] - probs[0]
                if (label == 1 and diff < 0) or (label == 0 and diff > 0):
                    results[aspect_label][label].append(1)
                    results[aspect_label][abs(label - 1)].append(-1)
                else:
                    results[aspect_label][label].append(0)
                    results[aspect_label][abs(label - 1)].append(0)

            if self.verbose:
                print(
                    f"=== Label: {label}, proba true:{probs[1]}, proba false:{probs[0]}, change: {results[aspect_label][label][-1]}"
                )

        return self.postprocess_results(results)

    def adapt_cav_intervention_vector(
        self, sentence, cav, intervention_direction="true2false"
    ):
        init_probs = self.get_probs(sentence, self.aspect_prompt)

        init_sum_of_probs = sum(init_probs)
        result_alpha = None
        does_change_class = False
        tot = 1

        alpha = -10
        tick = 10

        is_total_okey = True

        while is_total_okey and not does_change_class:
            alpha += tick

            if intervention_direction == "true2false":

                def hook_t2f(module, input, output):
                    output[0][:, -1, :] -= cav * alpha
                    return output

                handle = self.model.model.layers[self.layer].register_forward_hook(
                    hook_t2f
                )

            elif intervention_direction == "false2true":

                def hook_f2t(module, input, output):
                    output[0][:, -1, :] += cav * alpha
                    return output

                handle = self.model.model.layers[self.layer].register_forward_hook(
                    hook_f2t
                )

            probs = self.get_probs(sentence, self.aspect_prompt)
            handle.remove()

            proba_true = probs[1]
            proba_false = probs[0]
            # jak dużo się zmieniło czyli prawdopodobieństwo wygenerowania TRUE  - prawdopodobieństwo wygenerowania FALSE
            diff = proba_true - proba_false
            # liczymy też total prawd. dla true i false, bo może się zdarzyć, że zachce generowac całkiem inne słowa
            tot = proba_true + proba_false

            if tot > min(self.tot, init_sum_of_probs):
                is_total_okey = True
                if self.verbose:
                    print(
                        f"alpha: {alpha}, tick: {tick} diff: {diff}, tot: {tot}, proba_true: {proba_true}, proba_false: {proba_false}"
                    )

                if (intervention_direction == "true2false" and diff < 0) or (
                    intervention_direction == "false2true" and diff > 0
                ):
                    if tick <= self.tick:
                        result_alpha = alpha
                        does_change_class = True
                    else:
                        alpha -= tick
                        tick /= 2

            else:
                if tick > self.tick:
                    is_total_okey = True
                    alpha -= tick
                    tick /= 2
                else:
                    is_total_okey = False

        if result_alpha is None or result_alpha <= 0:
            return None

        sign = -1 if intervention_direction == "true2false" else 1

        return result_alpha * cav * sign

    def get_probs(self, sentence, prompt):
        input = prompt.format(sentence=sentence)
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        probs = self.model(input_ids).logits[0, -1, :].softmax(-1)
        return np.array([probs[token].item() for token in self.tokens])

    def postprocess_results(self, results):
        results = {
            0: {
                0: np.mean(results[0][0]) if len(results[0][0]) > 0 else 0,
                1: np.mean(results[0][1]) if len(results[0][1]) > 0 else 0,
            },
            1: {
                0: np.mean(results[1][0]) if len(results[1][0]) > 0 else 0,
                1: np.mean(results[1][1]) if len(results[1][1]) > 0 else 0,
            },
        }

        return results
