def load_prompt(data_dir, dataset_path, prompt_type, prompt_aspect):
    with open(
        f"{data_dir}/prompts/{prompt_type}/{dataset_path}/{prompt_aspect}.txt"
    ) as f:
        prompt = f.read()

    return prompt.strip()
