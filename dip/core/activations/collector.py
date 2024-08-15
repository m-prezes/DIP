import numpy as np
import torch as t
from master_thesis.core.activations.hook import attach_hooks, remove_hooks


class ActivationCollector:
    def __init__(self, model, tokenizer, layers, device):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device

    def get_acts(self, statements, prompt):
        """
        Get given layer activations for the statements.
        Return dictionary of stacked activations.
        """
        hooks, handles = attach_hooks(self.model, self.layers)

        answers = []
        acts = {layer: [] for layer in self.layers}
        for statement in statements:
            input = prompt.format(sentence=statement)
            input_ids = self.tokenizer.encode(input, return_tensors="pt").to(
                self.device
            )
            with t.no_grad():
                probs = self.model(input_ids)
            prediction = np.argmax(
                probs.logits[0, -1, :].softmax(-1).cpu().detach().numpy()
            )
            answers.append(
                self.postprocess_answer(
                    self.tokenizer.decode(prediction, skip_special_tokens=True)
                )
            )

            for layer, hook in zip(self.layers, hooks):
                acts[layer].append(hook.out[0, -1])

        for layer, act in acts.items():
            acts[layer] = t.stack(act).float()

        remove_hooks(handles)

        return acts, answers

    def postprocess_answer(self, answer):
        return answer.strip().lower()
