import json
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, mode='max', warmup=30):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.warmup    = warmup
        self.best      = -np.inf if mode == 'max' else np.inf
        self.counter   = 0
        self.triggered = False

    def step(self, value, epoch):
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

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def load_state_dict(self, d):
        self.__dict__.update(d)

    @property
    def status(self):
        return f"{self.counter}/{self.patience}  best={self.best:.4f}"


# ============================================================
# Overfitting Monitor
# ============================================================

class OverfitMonitor:
    def __init__(self, window=5, threshold=0.005):
        self.window    = window
        self.threshold = threshold
        self.history   = []

    def step(self, loss_value):
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

    def state_dict(self):
        return {'window': self.window, 'threshold': self.threshold,
                'history': self.history}

    def load_state_dict(self, d):
        self.window    = d['window']
        self.threshold = d['threshold']
        self.history   = d['history']


# ============================================================
# Plot helper
# ============================================================

def _save_training_plot(history: dict, out_path: str):
    epochs         = history['epoch']
    overfit_epochs = history.get('overfit_epochs', [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Training History  —  best dice {max(history['avg_dice']):.4f}",
        fontsize=14, fontweight='bold',
    )

    def _plot(ax, key, label, color, ylabel, title):
        ax.plot(epochs, history[key], color=color, linewidth=1.8, label=label)
        for ep in overfit_epochs:
            ax.axvline(ep, color='red', linewidth=0.6, alpha=0.4, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8)

    _plot(axes[0, 0], 'avg_loss',      'Total loss',  '#2563EB', 'Loss', 'Total Loss')
    _plot(axes[0, 1], 'avg_dice',      'Dice score',  '#16A34A', 'Dice', 'Dice Score  (higher = better)')
    _plot(axes[1, 0], 'avg_ce',        'CE loss',     '#9333EA', 'Loss', 'Cross-Entropy Loss')
    _plot(axes[1, 1], 'avg_dice_loss', 'Dice loss',   '#EA580C', 'Loss', 'Dice Loss')

    best_epoch = epochs[int(np.argmax(history['avg_dice']))]
    best_dice  = max(history['avg_dice'])
    axes[0, 1].axvline(best_epoch, color='#16A34A', linewidth=1.2, linestyle='-', alpha=0.8)
    axes[0, 1].annotate(
        f'best {best_dice:.4f}',
        xy=(best_epoch, best_dice),
        xytext=(8, -18), textcoords='offset points',
        fontsize=7, color='#16A34A',
        arrowprops=dict(arrowstyle='->', color='#16A34A', lw=0.8),
    )

    ax_lr = axes[0, 1].twinx()
    ax_lr.plot(epochs, history['lr'], color='gray', linewidth=0.9,
               linestyle=':', alpha=0.7, label='LR')
    ax_lr.set_ylabel('Learning Rate', color='gray', fontsize=8)
    ax_lr.tick_params(axis='y', labelcolor='gray', labelsize=7)
    ax_lr.legend(loc='lower right', fontsize=7)

    if overfit_epochs:
        from matplotlib.lines import Line2D
        legend_el = Line2D([0], [0], color='red', linewidth=0.8,
                           linestyle='--', alpha=0.6, label='overfit flag')
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles + [legend_el], labels + ['overfit flag'], fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Resume helpers
# ============================================================

RESUME_FILE = "resume.pth"   # fixed name — always lives in snapshot_path


def _save_resume(snapshot_path, epoch_num, iter_num,
                 model, optimizer, scheduler,
                 best_dice, best_model_path,
                 history, early_stopper, overfit_mon,
                 run_id, base_name):
    """
    Overwrites a single resume.pth every epoch.
    Contains everything needed to continue training exactly where it stopped.
    """
    payload = {
        'epoch_num'      : epoch_num,          # last *completed* epoch (0-based)
        'iter_num'       : iter_num,
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_dice'      : best_dice,
        'best_model_path': best_model_path,
        'history'        : history,
        'early_stopper'  : early_stopper.state_dict(),
        'overfit_mon'    : overfit_mon.state_dict(),
        'run_id'         : run_id,
        'base_name'      : base_name,
    }
    torch.save(payload, os.path.join(snapshot_path, RESUME_FILE))


def _load_resume(snapshot_path, model, optimizer, scheduler,
                 early_stopper, overfit_mon):
    """
    Looks for resume.pth in snapshot_path.
    If found, restores all states and returns a tuple so the trainer can
    pick up exactly where it left off.
    Returns None if no checkpoint exists (fresh run).
    """
    path = os.path.join(snapshot_path, RESUME_FILE)
    if not os.path.isfile(path):
        return None

    ckpt = torch.load(path, map_location='cpu')

    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    early_stopper.load_state_dict(ckpt['early_stopper'])
    overfit_mon.load_state_dict(ckpt['overfit_mon'])

    return (
        ckpt['epoch_num'] + 1,      # start_epoch: next epoch after the saved one
        ckpt['iter_num'],
        ckpt['best_dice'],
        ckpt['best_model_path'],
        ckpt['history'],
        ckpt['run_id'],
        ckpt['base_name'],
    )


# ============================================================
# Trainer
# ============================================================

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    # ------------------------------------------------------------------
    # Hyper-params
    # ------------------------------------------------------------------
    es_patience     = getattr(args, 'early_stop_patience', 20)
    es_warmup       = getattr(args, 'early_stop_warmup',   30)
    es_min_delta    = getattr(args, 'early_stop_delta',  1e-4)
    label_smoothing = getattr(args, 'label_smoothing',    0.1)
    overfit_window  = getattr(args, 'overfit_window',       5)

    base_lr     = args.base_lr
    num_classes = args.num_classes
    batch_size  = args.batch_size * args.n_gpu
    max_epoch   = args.max_epochs

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
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
    # Loss / optimiser / scheduler
    # These are built FIRST so _load_resume can overwrite their states.
    # ------------------------------------------------------------------
    ce_loss   = CrossEntropyLoss(label_smoothing=label_smoothing)
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epoch, eta_min=base_lr * 1e-3,
    )
    early_stopper = EarlyStopping(
        patience=es_patience, min_delta=es_min_delta,
        mode='max', warmup=es_warmup,
    )
    overfit_mon = OverfitMonitor(window=overfit_window)

    # ------------------------------------------------------------------
    # Resume detection — auto, no flag needed
    # ------------------------------------------------------------------
    resume_data = _load_resume(
        snapshot_path, model, optimizer, scheduler,
        early_stopper, overfit_mon,
    )

    if resume_data is not None:
        (start_epoch, iter_num, best_dice,
         best_model_path, history, run_id, base_name) = resume_data
        resumed = True
    else:
        start_epoch     = 0
        iter_num        = 0
        best_dice       = 0.0
        best_model_path = None
        run_id          = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name       = (
            f"bs{args.batch_size}"
            f"_skip{args.n_skip}"
            f"_ep{max_epoch}"
            f"_run{run_id}"
        )
        history = {
            'run_id'        : run_id,
            'base_name'     : base_name,
            'args'          : vars(args) if hasattr(args, '__dict__') else str(args),
            'epoch'         : [],
            'avg_loss'      : [],
            'avg_ce'        : [],
            'avg_dice_loss' : [],
            'avg_dice'      : [],
            'lr'            : [],
            'overfit_epochs': [],
        }
        resumed = False

    # ------------------------------------------------------------------
    # Logging  (file only — tqdm.write() is the sole console output)
    # ------------------------------------------------------------------
    # Root cause of double-prints: logging.info() fires both FileHandler
    # (disk) and StreamHandler (stdout), while tqdm.write() also writes
    # to stdout.  Fix: drop StreamHandler entirely; every line that should
    # appear on screen is passed through tqdm.write() explicitly.
    log_file = os.path.join(snapshot_path, f"log_{base_name}.txt")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(
        logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()   # prevent duplicate handlers on re-runs
    logger.addHandler(file_handler)

    def log(msg: str, console: bool = True):
        """Write to log file always; optionally echo to terminal via tqdm."""
        logging.info(msg)
        if console:
            # file=sys.stderr keeps tqdm bar and text on the same stream,
            # which is what makes the bar animate correctly in WSL terminals.
            tqdm.write(msg, file=sys.stderr)

    if resumed:
        banner = (
            f"\n{'>'*68}\n"
            f"  RESUMED  —  continuing from epoch {start_epoch}/{max_epoch}"
            f"  |  run_id={run_id}\n"
            f"  best_dice so far : {best_dice:.4f}  |  iter_num : {iter_num}\n"
            f"{'>'*68}"
        )
        log(banner)
    else:
        log(str(args), console=False)
        log(f"Run ID          : {run_id}")
        log(f"Base name       : {base_name}")
        log(
            f"Early stopping  : patience={es_patience} "
            f"warmup={es_warmup}  delta={es_min_delta}"
        )
        log(f"Label smoothing : {label_smoothing}")

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    writer = SummaryWriter(os.path.join(snapshot_path, f'log_{run_id}'))

    max_iterations = max_epoch * len(trainloader)
    log(
        f"{len(trainloader)} iters/epoch  |  {max_iterations} total  |  "
        f"starting at epoch {start_epoch + 1}"
    )

    # ------------------------------------------------------------------
    # Training loop
    # tqdm's initial= + total= keeps the bar accurate on resume
    # ------------------------------------------------------------------
    iterator = tqdm(
        range(start_epoch, max_epoch),
        initial=start_epoch,
        total=max_epoch,
        dynamic_ncols=True,   # adapts to terminal width (important in WSL)
        file=sys.stderr,      # stderr is unbuffered → bar animates in WSL
    )

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

            writer.add_scalar('iter/lr',         cur_lr,        iter_num)
            writer.add_scalar('iter/total_loss', loss,          iter_num)
            writer.add_scalar('iter/loss_ce',    loss_ce,       iter_num)
            writer.add_scalar('iter/loss_dice',  loss_dice,     iter_num)
            writer.add_scalar('iter/dice_score', 1 - loss_dice, iter_num)

            if iter_num % 100 == 0:
                log(
                    f"iter {iter_num:05d} | "
                    f"loss={loss.item():.4f}  "
                    f"ce={loss_ce.item():.4f}  "
                    f"dice={1 - loss_dice.item():.4f}  "
                    f"lr={cur_lr:.6f}",
                    console=False,   # iter-level noise stays in file only
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
        # LR scheduler step
        # --------------------------------------------------------------
        scheduler.step()

        # --------------------------------------------------------------
        # Epoch metrics
        # --------------------------------------------------------------
        avg_loss      = epoch_loss_sum / epoch_batches
        avg_ce        = epoch_ce_sum   / epoch_batches
        avg_dice_loss = epoch_dice_sum / epoch_batches
        avg_dice      = 1.0 - avg_dice_loss
        cur_lr        = optimizer.param_groups[0]['lr']

        writer.add_scalar('epoch/avg_loss',       avg_loss,      epoch_num)
        writer.add_scalar('epoch/avg_ce',         avg_ce,        epoch_num)
        writer.add_scalar('epoch/avg_dice_loss',  avg_dice_loss, epoch_num)
        writer.add_scalar('epoch/avg_dice_score', avg_dice,      epoch_num)
        writer.add_scalar('epoch/lr',             cur_lr,        epoch_num)

        history['epoch'].append(epoch_num + 1)
        history['avg_loss'].append(avg_loss)
        history['avg_ce'].append(avg_ce)
        history['avg_dice_loss'].append(avg_dice_loss)
        history['avg_dice'].append(avg_dice)
        history['lr'].append(cur_lr)

        # --------------------------------------------------------------
        # Overfitting check
        # --------------------------------------------------------------
        is_overfit, overfit_msg = overfit_mon.step(avg_loss)
        if is_overfit:
            history['overfit_epochs'].append(epoch_num + 1)

        # --------------------------------------------------------------
        # Epoch log — one line
        # --------------------------------------------------------------
        is_best = avg_dice > best_dice
        flags   = (" ★" if is_best else "") + (" ⚠" if is_overfit else "")
        ep_line = (
            f"epoch {epoch_num+1:03d}/{max_epoch}"
            f" | Loss: {avg_loss:.4f}"
            f" | CE: {avg_ce:.4f}"
            f" | Dice: {avg_dice:.4f}"
            f" | LR: {cur_lr:.2e}"
            f" | ES: {early_stopper.counter}/{es_patience}"
            + flags
        )
        log(ep_line)
        if is_overfit:
            log(f"         └─ ⚠  overfit — {overfit_msg}")

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
                f"BEST_{base_name}_epoch{epoch_num+1:03d}_dice{best_dice:.4f}.pth"
            )
            torch.save(model.state_dict(), best_model_path)
            log(f"  ★ Best model → {best_model_path}")

        # --------------------------------------------------------------
        # Checkpoint every 5 epochs
        # --------------------------------------------------------------
        if (epoch_num + 1) % 5 == 0:
            ckpt_path = os.path.join(
                snapshot_path,
                f"ckpt_{base_name}_epoch{epoch_num+1:03d}_dice{avg_dice:.4f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            log(f"  ✔ Checkpoint → {ckpt_path}")

        # --------------------------------------------------------------
        # ★ Save resume checkpoint  (same file, overwritten every epoch)
        # --------------------------------------------------------------
        _save_resume(
            snapshot_path, epoch_num, iter_num,
            model, optimizer, scheduler,
            best_dice, best_model_path,
            history, early_stopper, overfit_mon,
            run_id, base_name,
        )

        # --------------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------------
        if early_stopper.step(avg_dice, epoch_num):
            msg = (
                f"\n{'!'*68}\n"
                f"  EARLY STOPPING — epoch {epoch_num+1}\n"
                f"  No dice improvement for {es_patience} epochs "
                f"(warmup={es_warmup})\n"
                f"  Best dice  : {best_dice:.4f}\n"
                f"  Best model : {best_model_path}\n"
                f"{'!'*68}"
            )
            log(msg)
            iterator.close()
            break

        # --------------------------------------------------------------
        # Final epoch
        # --------------------------------------------------------------
        if epoch_num >= max_epoch - 1:
            final_path = os.path.join(
                snapshot_path,
                f"final_{base_name}_epoch{epoch_num+1:03d}_dice{avg_dice:.4f}.pth"
            )
            torch.save(model.state_dict(), final_path)
            log(f"  Final model → {final_path}")
            iterator.close()
            break

    writer.close()

    # ------------------------------------------------------------------
    # Save history JSON
    # ------------------------------------------------------------------
    history_path = os.path.join(snapshot_path, f"history_{base_name}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log(f"  History JSON → {history_path}")

    # ------------------------------------------------------------------
    # Save training plot
    # ------------------------------------------------------------------
    plot_path = os.path.join(snapshot_path, f"plot_{base_name}.png")
    _save_training_plot(history, plot_path)
    log(f"  Training plot → {plot_path}")

    # ------------------------------------------------------------------
    # Last-epoch model
    # ------------------------------------------------------------------
    last_path = os.path.join(
        snapshot_path, f"last_{base_name}_epoch{epoch_num+1:03d}"
    )
    torch.save(model.state_dict(), last_path)
    log(f"  Last model → {last_path}")

    log(
        f"\nDone.  Best dice={best_dice:.4f}  "
        f"Model={best_model_path}  Run={run_id}"
    )
    iterator.close()
    return "Training Finished!"