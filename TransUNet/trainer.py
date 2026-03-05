import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Stops training when the monitored metric stops improving.

    Args:
        patience  (int)   : epochs to wait after last improvement
        min_delta (float) : minimum change to qualify as improvement
        mode      (str)   : 'max' for dice, 'min' for loss
        warmup    (int)   : epochs to skip before activating
    """

    def __init__(self, patience=20, min_delta=1e-4, mode='max', warmup=30):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.warmup    = warmup
        self.best      = -np.inf if mode == 'max' else np.inf
        self.counter   = 0
        self.triggered = False

    def step(self, value, epoch):
        """Returns True if training should stop."""
        if epoch < self.warmup:
            return False

        improved = (
            value > self.best + self.min_delta if self.mode == 'max'
            else value < self.best - self.min_delta
        )

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.triggered = True
            return True
        return False

    @property
    def status(self):
        return f"{self.counter}/{self.patience}  best={self.best:.4f}"


# ============================================================
# Overfitting Monitor
# ============================================================

class OverfitMonitor:
    """
    Watches the rolling slope of the training loss.
    A consistently rising loss after initial convergence
    is a strong signal of overfitting.

    Warns in the log but does NOT stop training — pair it
    with EarlyStopping for automatic stopping.
    """

    def __init__(self, window=5, threshold=0.005):
        self.window    = window
        self.threshold = threshold
        self.history   = []

    def step(self, loss_value):
        """Returns (bool: overfitting, str: message)."""
        self.history.append(loss_value)
        if len(self.history) < self.window:
            return False, ""

        recent = self.history[-self.window:]
        slope  = np.polyfit(np.arange(self.window, dtype=float), recent, 1)[0]

        if slope > self.threshold:
            return True, (
                f"loss rising over last {self.window} epochs "
                f"(slope={slope:+.4f})"
            )
        return False, ""


# ============================================================
# Trainer
# ============================================================

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    # ------------------------------------------------------------------
    # Anti-overfitting hyper-params — read from args with safe defaults
    # Override by adding these to your train.py argparse if needed:
    #   --early_stop_patience 20
    #   --early_stop_warmup   30
    #   --early_stop_delta    0.0001
    #   --label_smoothing     0.1
    # ------------------------------------------------------------------
    es_patience     = getattr(args, 'early_stop_patience', 20)
    es_warmup       = getattr(args, 'early_stop_warmup',   30)
    es_min_delta    = getattr(args, 'early_stop_delta',  1e-4)
    label_smoothing = getattr(args, 'label_smoothing',    0.1)
    overfit_window  = getattr(args, 'overfit_window',       5)

    # ------------------------------------------------------------------
    # Run ID and checkpoint base name
    # ------------------------------------------------------------------
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        f"bs{args.batch_size}"
        f"_skip{args.n_skip}"
        f"_ep{args.max_epochs}"
        f"_run{run_id}"
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_file = os.path.join(snapshot_path, f"log_{base_name}.txt")
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"Run ID           : {run_id}")
    logging.info(f"Base name        : {base_name}")
    logging.info(
        f"Early stopping   : patience={es_patience}  "
        f"warmup={es_warmup}  delta={es_min_delta}"
    )
    logging.info(f"Label smoothing  : {label_smoothing}")

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    base_lr     = args.base_lr
    num_classes = args.num_classes
    batch_size  = args.batch_size * args.n_gpu

    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ),
    )
    print(f"Train set size : {len(db_train)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # ------------------------------------------------------------------
    # Loss functions
    # label_smoothing on CE reduces over-confident predictions,
    # which is one of the earliest signs of overfitting.
    # ------------------------------------------------------------------
    ce_loss   = CrossEntropyLoss(label_smoothing=label_smoothing)
    dice_loss = DiceLoss(num_classes)

    # ------------------------------------------------------------------
    # Optimiser
    # weight_decay=1e-4 acts as L2 regularisation → penalises large weights
    # ------------------------------------------------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr, momentum=0.9, weight_decay=1e-4,
    )

    max_epoch = args.max_epochs

    # Cosine annealing LR — decays smoothly to near-zero, then can
    # warm back up.  Much gentler than poly decay on long runs and
    # helps avoid sharp loss spikes that precede overfitting.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=base_lr * 1e-3
    )

    # ------------------------------------------------------------------
    # Overfitting guards
    # ------------------------------------------------------------------
    early_stopper = EarlyStopping(
        patience=es_patience, min_delta=es_min_delta,
        mode='max', warmup=es_warmup,
    )
    overfit_mon = OverfitMonitor(window=overfit_window)

    writer = SummaryWriter(os.path.join(snapshot_path, f'log_{run_id}'))

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    iter_num        = 0
    max_iterations  = max_epoch * len(trainloader)
    best_dice       = 0.0
    best_model_path = None

    logging.info(
        f"{len(trainloader)} iters/epoch  |  "
        f"{max_iterations} total iters"
    )

    iterator = tqdm(range(max_epoch), ncols=90)

    for epoch_num in iterator:

        model.train()
        epoch_loss_sum = epoch_ce_sum = epoch_dice_sum = 0.0
        epoch_batches  = 0

        # --------------------------------------------------------------
        # Batch loop
        # --------------------------------------------------------------
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            outputs   = model(image_batch)
            loss_ce   = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss      = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num       += 1
            epoch_loss_sum += loss.item()
            epoch_ce_sum   += loss_ce.item()
            epoch_dice_sum += loss_dice.item()
            epoch_batches  += 1

            cur_lr = optimizer.param_groups[0]['lr']

            # TensorBoard per-iteration
            writer.add_scalar('iter/lr',         cur_lr,          iter_num)
            writer.add_scalar('iter/total_loss', loss,            iter_num)
            writer.add_scalar('iter/loss_ce',    loss_ce,         iter_num)
            writer.add_scalar('iter/loss_dice',  loss_dice,       iter_num)
            writer.add_scalar('iter/dice_score', 1 - loss_dice,   iter_num)
            if iter_num % 100 == 0:
                logging.info(
                    f"iter {iter_num:05d} | "
                    f"loss={loss.item():.4f}  "
                    f"ce={loss_ce.item():.4f}  "
                    f"dice={1 - loss_dice.item():.4f}  "
                    f"lr={cur_lr:.6f}"
                )

            if iter_num % 20 == 0:
                img = image_batch[1, 0:1]
                img = (img - img.min()) / (img.max() - img.min())
                writer.add_image('train/Image', img, iter_num)
                pred = torch.argmax(
                    torch.softmax(outputs, dim=1), dim=1, keepdim=True
                )
                writer.add_image('train/Prediction', pred[1] * 50, iter_num)
                writer.add_image(
                    'train/GroundTruth',
                    label_batch[1].unsqueeze(0) * 50, iter_num,
                )

        # --------------------------------------------------------------
        # Step LR scheduler (end of epoch)
        # --------------------------------------------------------------
        scheduler.step()

        # --------------------------------------------------------------
        # Epoch metrics
        # --------------------------------------------------------------
        avg_loss      = epoch_loss_sum / epoch_batches
        avg_ce        = epoch_ce_sum   / epoch_batches
        avg_dice_loss = epoch_dice_sum / epoch_batches
        avg_dice      = 1.0 - avg_dice_loss

        writer.add_scalar('epoch/avg_loss',       avg_loss,      epoch_num)
        writer.add_scalar('epoch/avg_ce',         avg_ce,        epoch_num)
        writer.add_scalar('epoch/avg_dice_loss',  avg_dice_loss, epoch_num)
        writer.add_scalar('epoch/avg_dice_score', avg_dice,      epoch_num)
        writer.add_scalar('epoch/lr',
                          optimizer.param_groups[0]['lr'], epoch_num)

        # --------------------------------------------------------------
        # Overfitting check (warning only)
        # --------------------------------------------------------------
        is_overfit, overfit_msg = overfit_mon.step(avg_loss)

        # --------------------------------------------------------------
        # Epoch summary
        # --------------------------------------------------------------
        is_best = avg_dice > best_dice
        logging.info(
            f"\n{'='*68}\n"
            f"  Epoch {epoch_num+1:03d}/{max_epoch}  |  Run: {run_id}\n"
            f"  avg_loss      : {avg_loss:.4f}\n"
            f"  avg_ce_loss   : {avg_ce:.4f}\n"
            f"  avg_dice_loss : {avg_dice_loss:.4f}\n"
            f"  avg_dice      : {avg_dice:.4f}"
            f"{'  ★ NEW BEST' if is_best else ''}\n"
            f"  lr            : {optimizer.param_groups[0]['lr']:.6f}\n"
            f"  Early stop    : {early_stopper.status}\n"
            + (f"  ⚠  OVERFIT WARNING — {overfit_msg}\n" if is_overfit else "")
            + f"{'='*68}"
        )

        iterator.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'dice': f'{avg_dice:.3f}',
            'pat' : f'{early_stopper.counter}/{es_patience}',
        })

        # --------------------------------------------------------------
        # Save best model
        # --------------------------------------------------------------
        if is_best:
            best_dice = avg_dice
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(
                snapshot_path,
                f"BEST_{base_name}_epoch{epoch_num+1:03d}"
                f"_dice{best_dice:.4f}.pth"
            )
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  ★ Best model → {best_model_path}")

        # --------------------------------------------------------------
        # Checkpoint every 5 epochs
        # --------------------------------------------------------------
        if (epoch_num + 1) % 5 == 0:
            ckpt_path = os.path.join(
                snapshot_path,
                f"ckpt_{base_name}_epoch{epoch_num+1:03d}"
                f"_dice{avg_dice:.4f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"  ✔ Checkpoint → {ckpt_path}")

        # --------------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------------
        if early_stopper.step(avg_dice, epoch_num):
            logging.info(
                f"\n{'!'*68}\n"
                f"  EARLY STOPPING  —  epoch {epoch_num+1}\n"
                f"  No dice improvement for {es_patience} epochs "
                f"(warmup={es_warmup})\n"
                f"  Best dice  : {best_dice:.4f}\n"
                f"  Best model : {best_model_path}\n"
                f"{'!'*68}"
            )
            iterator.close()
            break

        # --------------------------------------------------------------
        # Final epoch
        # --------------------------------------------------------------
        if epoch_num >= max_epoch - 1:
            final_path = os.path.join(
                snapshot_path,
                f"final_{base_name}_epoch{epoch_num+1:03d}"
                f"_dice{avg_dice:.4f}.pth"
            )
            torch.save(model.state_dict(), final_path)
            logging.info(f"  ✔ Final model → {final_path}")
            iterator.close()
            break

    writer.close()
    logging.info(
        f"\nDone.  Best dice={best_dice:.4f}  "
        f"Model={best_model_path}  Run={run_id}"
    )
    return "Training Finished!"