import numpy as np
from tqdm import tqdm


def evaluate_model_accuracy(
    model, tokenizer, dataset, prompt, label_name, tokens, device
):
    acc = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        sentence = row["sentence"]
        label = row[label_name]

        input_text = prompt.format(sentence=sentence)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        probs = model(input_ids).logits[0, -1, :].softmax(-1)

        prediction = np.argmax(probs.cpu().detach().numpy())

        acc.append(prediction == tokens[label])

    return np.mean(acc)
