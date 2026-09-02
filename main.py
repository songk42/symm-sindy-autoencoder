import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from functools import partial
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from cmd_line import parse_args
from src.trainer.baseline import train, test
from src.utils.other import *#load_data, load_model, make_model, get_tb_path, get_checkpoint_path, get_args_path, get_experiments_path
from src.utils.model_utils import init_weights


_METRIC_ABBR = {
    'L recon': 'recon', 'L dx': 'dx', 'L dz': 'dz', 'L regularization': 'reg',
    'KLD': 'kld',
}


def _fmt_metrics(split, epoch, metrics):
    body = "  ".join(f"{_METRIC_ABBR.get(k, k)}={v:.3e}" for k, v in metrics.items())
    return f"[epoch {epoch:4d}] {split:5s} {body}"


def main():
    # get and save args
    args = parse_args()

    # train and val data (will refer to the val data as test data)
    train_set, test_set, _ = load_data(args)

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # experiment logging (wandb)
    run_name = get_general_path(args).strip('/').replace('/', '_')
    wandb_on = init_wandb(args, run_name)
    train_log = partial(log_metrics, 'train', enabled=wandb_on)
    test_log = partial(log_metrics, 'val', enabled=wandb_on)

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    # checkpoint, args, experiments path
    cp_path, cp_folder = get_checkpoint_path(args)
    args_path, args_folder = get_args_path(args)
    exp_folder = get_experiments_path(args)
    if not os.path.isdir(cp_folder):
        os.system("mkdir -p " + cp_folder)
    if not os.path.isdir(exp_folder):
        os.system("mkdir -p " + exp_folder)
    if args.print_folder == 1:
        print("Checkpoints saved at:        ", cp_folder)
        print("Experiment results saved at: ", exp_folder)
        print("wandb logging:               ", args.wandb_mode if wandb_on else "off")

    # save args
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # create model, optim, scheduler, initial epoch
    net = make_model(args).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.adam_regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
    initial_e = 0
    
    # load model, optim, scheduler, epoch from checkpoint
    if args.load_cp == 1:
        net, optim, scheduler, initial_e = load_model(net, cp_path, device, optim, scheduler)
    else:  # init network
        net.apply(init_weights)

    # lambdas for loss function
    lambdas = args.lambda_dx, args.lambda_dz, args.lambda_reg

    # for each epoch
    for epoch in tqdm(range(args.epochs), desc="Epoch", total=args.epochs, dynamic_ncols=True):
        # train
        train_metrics = train(net, train_loader, train_log, optim, epoch + initial_e, args.clip, lambdas)
        tqdm.write(_fmt_metrics('train', epoch + initial_e, train_metrics))

        # test
        if (epoch + 1) % args.test_interval == 0:
            test_metrics = test(net, test_loader, test_log, epoch + initial_e, args.timesteps, lambdas)
            tqdm.write(_fmt_metrics('val', epoch + initial_e, test_metrics))

        # step on learning rate scheduler
        scheduler.step()
    
        # save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint = {'epoch': epoch + initial_e,
                          'model': net.state_dict(),
                          'optimizer': optim.state_dict(),
                          'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, cp_path)

    if wandb_on:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()