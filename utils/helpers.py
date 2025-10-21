import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torchvision.utils


def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )
def gridify_output_with_annotations(img, row_size=-1, names = None, output_name = "res"):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    grid = torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0)

    # Calculate the number of rows needed for the grid
    num_images = img.shape[0]
    if row_size == -1:
        row_size = num_images
    num_rows = (num_images - 1) // row_size + 1

    # Determine the size of each image in the grid
    grid_height, grid_width, _ = grid.shape
    image_height = grid_height // num_rows
    image_width = grid_width // row_size

    # Create the plot and add the grid of images
    plt.figure(figsize=(row_size * 3, num_rows * 3))
    plt.imshow(grid, cmap = 'gray')

    # Add the name as text in the top-left corner of each image
    for i in range(num_images):
        row_idx = i // row_size
        col_idx = i % row_size
        if names is not None:
            image_name = names[i]
        else:
            image_name = f"Image {i+1}"
        text_x = (col_idx * image_width) + (image_width * 0.02)  # Adjust the 0.02 value to change the horizontal spacing of the text
        text_y = (row_idx * image_height) + (image_height * 0.02)  # Adjust the 0.02 value to change the vertical spacing of the text
        plt.text(text_x, text_y, image_name, fontsize=12, ha='left', va='top', color='white')
 
    plt.axis('off')
    plt.savefig(output_name + ".png")
    plt.close('all')


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


def load_checkpoint(param, use_checkpoint, device):
    """
    loads the most recent (non-corrupted) checkpoint or the final model
    :param param: args number
    :param use_checkpoint: checkpointed or final model
    :return:
    """
    # Resolve the actual model directory and arg number from various accepted param formats
    def _resolve_model_dir(p):
        base = './model'
        # If already a subdir under ./model
        if isinstance(p, str) and os.path.isdir(os.path.join(base, p)):
            model_dir = os.path.join(base, p)
        # If full or relative path provided
        elif isinstance(p, str) and os.path.isdir(p):
            model_dir = p
        else:
            # Normalize inputs like '26', 'args26', 'args26.json', 'diff-params-ARGS=26'
            if isinstance(p, str) and p.startswith('diff-params-ARGS='):
                num = p.split('=')[-1]
            elif isinstance(p, str) and p.startswith('args') and p.endswith('.json'):
                num = p[4:-5]
            elif isinstance(p, str) and p.startswith('args'):
                num = p[4:]
            else:
                num = str(p)
            model_dir = os.path.join(base, f'diff-params-ARGS={num}')

        # Derive arg number from directory name
        name = os.path.basename(model_dir.rstrip('/'))
        if 'diff-params-ARGS=' in name:
            arg_num = name.split('=')[-1]
        else:
            import re
            m = re.search(r'(\d+)', str(p))
            arg_num = m.group(1) if m else ''
        return model_dir, arg_num

    model_dir, _ = _resolve_model_dir(param)
    if not os.path.isdir(model_dir):
        available = []
        try:
            available = sorted([d for d in os.listdir('./model') if os.path.isdir(os.path.join('./model', d))])
        except Exception:
            pass
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Available: {available}")

    if not use_checkpoint:
        # Try common locations for the final params
        candidates = [
            os.path.join(model_dir, 'params-final.pt'),
            os.path.join(model_dir, 'checkpoint', 'params-final.pt'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return torch.load(path, map_location=device)
        # Fallback to most recent checkpoint if final not found
        use_checkpoint = True

    # Load from checkpoints directory
    ckpt_dir = os.path.join(model_dir, 'checkpoint')
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    entries = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not entries:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    # Prefer params-final.pt if present, otherwise newest (reverse lexicographic)
    ordered = ([] if 'params-final.pt' not in entries else ['params-final.pt']) + \
              sorted([f for f in entries if f != 'params-final.pt'], reverse=True)

    last_exc = None
    for fname in ordered:
        try:
            return torch.load(os.path.join(ckpt_dir, fname), map_location=device)
        except RuntimeError as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    raise FileNotFoundError(f"Unable to load a valid checkpoint from: {ckpt_dir}")


def load_parameters(device, argN = None):
    """
    Loads the trained parameters for the detection model
    :return:
    """
    import sys
    if argN is not None:
        print(f"Loading from {argN} Number")
#         params = os.listdir("./model")
        # Use a list comprehension to find the matching string
        params = [f'{argN}'] # [s for s in params if f'={argN}' in s]

    elif len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = os.listdir("./model")
    if ".DS_Store" in params:
        params.remove(".DS_Store")

    if params[0] == "CHECKPOINT":
        use_checkpoint = True
        params = params[1:]
    else:
        use_checkpoint = False
#     use_checkpoint = True
#     print(params)
    for param in params:
        # Normalize param to support multiple formats
        if isinstance(param, str) and param.isnumeric():
            norm = param
        elif isinstance(param, str) and param.startswith("args") and param.endswith(".json"):
            norm = param[4:-5]
        elif isinstance(param, str) and param.startswith("args"):
            norm = param[4:]
        elif isinstance(param, str) and param.startswith("diff-params-ARGS="):
            norm = param  # already a model folder name
        else:
            norm = str(param)

        output = load_checkpoint(norm, use_checkpoint, device)

        if "args" in output:
            args = output["args"]
        else:
            # Derive the argument number robustly for loading args JSON
            arg_num = None
            if isinstance(param, str) and param.startswith('diff-params-ARGS='):
                arg_num = param.split('=')[-1]
            elif isinstance(param, str) and param.startswith('args') and param.endswith('.json'):
                arg_num = param[4:-5]
            elif isinstance(param, str) and param.startswith('args'):
                arg_num = param[4:]
            elif isinstance(param, str) and param.isnumeric():
                arg_num = param
            else:
                import re
                m = re.search(r'(\d+)', str(param))
                arg_num = m.group(1) if m else None

            if not arg_num:
                raise ValueError(f"Could not infer arg number from parameter '{param}'")

            try:
                with open(f'./test_args/args{arg_num}.json', 'r') as f:
                    args = json.load(f)
                args['arg_num'] = arg_num
                args = defaultdict_from_json(args)
            except FileNotFoundError:
                raise ValueError(f"args{arg_num} doesn't exist for {param}")

        if "noise_fn" not in args:
            args["noise_fn"] = "gauss"

        return args, output


def main():
    pass


if __name__ == '__main__':
    main()
