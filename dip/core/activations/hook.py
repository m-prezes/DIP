class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs


def attach_hooks(model, layers):
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    return hooks, handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()
