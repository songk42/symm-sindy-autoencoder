import torch

from src.dataset.Datasets import *


def get_device(args):
    """Resolve args.device to a torch.device.

    Accepts a CUDA index (int, or numeric string, e.g. 0), "cpu", "mps", or a
    negative index (falls back to CPU) -- so runs work on machines without an
    NVIDIA GPU (e.g. Apple Silicon via "mps", or "cpu" anywhere).
    """
    d = getattr(args, "device", -1)
    if isinstance(d, str):
        d = d.strip().lower()
        if d in ("cpu", "mps"):
            return torch.device(d)
        d = int(d)
    if d < 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{d}")


def log_metrics(prefix, metrics, step, enabled=True):
    """Log a dict of scalars to wandb under a "<prefix>/" namespace.

    A no-op when wandb logging is disabled / not initialised. Bind the first
    two args with functools.partial to hand the trainer a simple callable:
        train_log = partial(log_metrics, 'train', enabled=wandb_on)
        train_log({'L recon': ...}, epoch)
    """
    if not enabled:
        return
    import wandb
    if wandb.run is None:
        return
    wandb.log({f"{prefix}/{tag}": value for tag, value in metrics.items()},
              step=int(step))


def init_wandb(args, run_name):
    """Start a wandb run according to args.wandb_mode. Returns True if logging
    is active. Missing wandb or mode='disabled' -> returns False, no crash."""
    mode = getattr(args, 'wandb_mode', 'online')
    if mode == 'disabled':
        return False
    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed; running without experiment logging")
        return False
    tags = None
    if getattr(args, 'wandb_tags', None):
        tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.data_set,
        mode=mode,
        tags=tags,
        config=vars(args),
    )
    return True


def load_data(args):
    # train and val data (using val as "test" data)
    if args.data_set == "lorenz":
        folder, data_paths = get_lorenz_path()
        train_set = LorenzDataset(args, data_paths[0])
        val_set = LorenzDataset(args, data_paths[1])
        test_set = LorenzDataset(args, data_paths[2])
    elif args.data_set == "synth":
        folder, data_paths = get_synth_path()
        train_set = SynthDataset(args, data_paths[0])
        val_set = SynthDataset(args, data_paths[1])
        test_set = SynthDataset(args, data_paths[2])
    elif args.data_set == "lorenz_distort":
        folder, data_paths = get_lorenz_distort_path()
        train_set = SynthDataset(args, data_paths[0])
        val_set = SynthDataset(args, data_paths[1])
        test_set = SynthDataset(args, data_paths[2])
    elif args.data_set == "cylinder":
        folder, data_paths = get_cylinder_path()
        train_set = SynthDataset(args, data_paths[0])
        val_set = SynthDataset(args, data_paths[1])
        test_set = SynthDataset(args, data_paths[2])
    return train_set, val_set, test_set

def load_model(net, cp_path, device, optim=None, scheduler=None):
    checkpoint = torch.load(cp_path, map_location=device)
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    initial_e = checkpoint['epoch']
    return net, optim, scheduler, initial_e

def make_model(args):
    if args.model == 'SINDyAE':
        from src.models.SINDyAE import Net
    if args.model == 'SINDyVAE':
        from src.models.SINDyVAE import Net
    return Net(args)

def get_lorenz_path():
    folder = "data/lorenz/"
    return folder, (folder + "train.npy", folder + "val.npy", folder + "test.npy")

def get_synth_path():
    folder = "data/synth/"
    return folder, (folder + "train.npy", folder + "val.npy", folder + "test.npy")

def get_lorenz_distort_path():
    folder = "data/lorenz_distort/"
    return folder, (folder + "train.npy", folder + "val.npy", folder + "test.npy")

def get_cylinder_path():
    folder = "data/cylinder/"
    return folder, (folder + "train.npy", folder + "val.npy", folder + "test.npy")

def get_general_path(args):
    return args.data_set + "/" + args.model + "/" + args.session_name + "/"

def get_checkpoint_path(args):
    cp_folder = args.model_folder + get_general_path(args)
    return cp_folder + 'checkpoint.pt', cp_folder

def get_args_path(args):
    args_folder = args.model_folder +get_general_path(args)
    return args_folder + "args.txt", args_folder

def get_tb_path(args):
    train_name = args.tensorboard_folder + get_general_path(args) + "train"
    test_name = args.tensorboard_folder + get_general_path(args) + "val"
    return train_name, test_name

def get_experiments_path(args):
    return args.experiments + get_general_path(args)