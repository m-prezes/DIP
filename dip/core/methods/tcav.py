import numpy as np
from tqdm import tqdm


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        module_outputs[0].retain_grad()
        self.out, _ = module_outputs


class TCAV:
    def __init__(
        self,
        cav,
        model,
        tokenizer,
        prompt,
        layer,
        tokens=list[int],
        device="cpu",
        verbose=False,
    ):
        self.device = device
        self.cav = cav.detach().cpu() / np.linalg.norm(cav.detach().cpu())
        self.tokens = tokens
        self.verbose = verbose
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.prompt = prompt

    def evaluate(self, dataset, aspect_label_name):
        labels = dataset[aspect_label_name].to_numpy()
        sensitivities = self.sensitivities(dataset, aspect_label_name)

        results = {
            0: {
                0: np.mean(sensitivities[labels == 0][:, 0] > 0),
                1: np.mean(sensitivities[labels == 1][:, 0] > 0),
            },
            1: {
                0: np.mean(sensitivities[labels == 0][:, 1] > 0),
                1: np.mean(sensitivities[labels == 1][:, 1] > 0),
            },
        }
        return results

    def sensitivities(self, dataset, aspect):
        scores = []
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            label = row[aspect]
            if label == -1:
                continue
            sentence = row["sentence"]

            grad_false_token = self.get_gradient(sentence, self.tokens[0])
            grad_false_token /= np.linalg.norm(grad_false_token)
            grad_true_token = self.get_gradient(sentence, self.tokens[1])
            grad_true_token /= np.linalg.norm(grad_true_token)

            cavs_scores = []
            cav = self.cav if label == 1 else -self.cav
            for grad in [grad_false_token, grad_true_token]:
                print(grad)
                print(cav)
                cavs_scores.append(np.dot(cav, grad))

            scores.append(np.array(cavs_scores))
            if self.verbose:
                print(f"Label: {label}, scores: {cavs_scores}")
        return np.array(scores)

    def get_gradient(self, sentence, token):
        hook = Hook()
        handle = self.model.model.layers[self.layer].register_forward_hook(hook)

        input_ids = self.tokenizer.encode(
            self.prompt.format(sentence=sentence), return_tensors="pt"
        ).to(self.device)
        probs = self.model(input_ids).logits[0, -1, :]
        self.model.zero_grad()
        probs[token].backward()
        grad = hook.out.grad.detach().cpu().numpy()
        result_grad = grad[0, -1].copy()
        hook.out.grad.zero_()

        handle.remove()
        return result_grad
