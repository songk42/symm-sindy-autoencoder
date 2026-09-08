import torch


def _run(net, loader, lambdas, optim=None, clip=None):
    """One pass over `loader`. Trains if `optim` is given, else evaluates.
    Returns a dict of epoch-averaged loss terms."""
    train_mode = optim is not None
    net.train() if train_mode else net.eval()

    totals = {k: 0.0 for k in ('recon', 'dx', 'dz', 'reg', 'gauge', 'equiv', 'jac', 'kl')}
    desc = "Training" if train_mode else "Testing"
    for x, dx, dz in loader:
        l_recon, l_dx, l_dz, l_reg, l_gauge, l_equiv, l_jac, kl = net(x, dx, lambdas)
        for k, v in zip(totals, (l_recon, l_dx, l_dz, l_reg, l_gauge, l_equiv, l_jac, kl)):
            totals[k] += v.item()

        if train_mode:
            batch_loss = (l_recon + l_dx + l_dz + l_reg + l_gauge + l_equiv + l_jac + kl) / len(x)
            optim.zero_grad()
            batch_loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optim.step()
            # sequential thresholding: freeze coefficients below the threshold
            net.threshold_mask[torch.abs(net.sindy_coefficients) < net.sequential_threshold] = 0

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def _metrics(avg):
    return {
        'L recon': avg['recon'],
        'L dx': avg['dx'],
        'L dz': avg['dz'],
        'L regularization': avg['reg'],
        'L gauge': avg['gauge'],
        'L equiv': avg['equiv'],
        'L jac': avg['jac'],
        'KLD': avg['kl'],
    }


def train(net, train_loader, log, optim, epoch, clip, lambdas):
    metrics = _metrics(_run(net, train_loader, lambdas, optim=optim, clip=clip))
    log(metrics, epoch)
    return metrics


def test(net, test_loader, log, epoch, timesteps, lambdas):
    with torch.no_grad():
        metrics = _metrics(_run(net, test_loader, lambdas))
    log(metrics, epoch)
    return metrics
