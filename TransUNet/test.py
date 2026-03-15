import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path',      type=str,   default='./data/Synapse/test_vol_h5')
parser.add_argument('--dataset',          type=str,   default='Synapse')
parser.add_argument('--num_classes',      type=int,   default=9)
parser.add_argument('--list_dir',         type=str,   default='./lists/lists_Synapse')
parser.add_argument('--max_iterations',   type=int,   default=20000)
parser.add_argument('--max_epochs',       type=int,   default=150)
parser.add_argument('--batch_size',       type=int,   default=24)
parser.add_argument('--img_size',         type=int,   default=224)
parser.add_argument('--is_savenii',       action="store_true")
parser.add_argument('--n_skip',           type=int,   default=3)
parser.add_argument('--vit_name',         type=str,   default='R50-ViT-B_16')
parser.add_argument('--test_save_dir',    type=str,   default='../predictions')
parser.add_argument('--deterministic',    type=int,   default=1)
parser.add_argument('--base_lr',          type=float, default=0.01)
parser.add_argument('--seed',             type=int,   default=1234)
parser.add_argument('--vit_patches_size', type=int,   default=16)

# Accept either:
#   (a) a direct path to a .pth file  e.g. .../BEST_xxx.pth
#   (b) a folder containing checkpoints
#   (c) empty string -> auto-build folder path from other args
parser.add_argument('--snapshot_path',    type=str,   default='',
                    help='Path to a .pth checkpoint file OR a snapshot folder. '
                         'Leave empty to auto-build from args.')

args = parser.parse_args()


# -----------------------------------------------------------------------
# Find the best checkpoint inside a folder.
# Priority:
#   1. Any file starting with BEST_   (custom trainer naming)
#   2. Any file starting with final_
#   3. Any file starting with ckpt_   (latest by name)
#   4. best_model.pth                 (original TransUNet name)
#   5. epoch_{max_epochs-1}.pth       (original TransUNet fallback)
# -----------------------------------------------------------------------
def find_checkpoint(folder, max_epochs):
    def glob_pick(prefix):
        matches = sorted([
            f for f in os.listdir(folder)
            if f.startswith(prefix) and f.endswith('.pth')
        ])
        return os.path.join(folder, matches[-1]) if matches else None

    for prefix in ('BEST_', 'final_', 'ckpt_'):
        path = glob_pick(prefix)
        if path:
            return path

    for name in ('best_model.pth', f'epoch_{max_epochs - 1}.pth'):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path

    return None


def inference(args, model, test_save_path=None):
    db_test    = args.Dataset(
        base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test cases")

    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image     = sampled_batch["image"]
        label     = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]

        metric_i = test_single_volume(
            image, label, model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        metric_list += np.array(metric_i)
        logging.info(
            f"idx {i_batch}  case {case_name}  "
            f"mean_dice {np.mean(metric_i, axis=0)[0]:.4f}  "
            f"mean_hd95 {np.mean(metric_i, axis=0)[1]:.4f}"
        )

    metric_list = metric_list / len(db_test)

    logging.info("\n" + "="*55)
    for i in range(1, args.num_classes):
        logging.info(
            f"  Class {i:02d} | dice {metric_list[i-1][0]:.4f} | "
            f"hd95 {metric_list[i-1][1]:.4f}"
        )
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95   = np.mean(metric_list, axis=0)[1]
    logging.info("="*55)
    logging.info(f"  MEAN   | dice {performance:.4f} | hd95 {mean_hd95:.4f}")
    logging.info("="*55)
    return "Testing Finished!"


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    if not args.deterministic:
        cudnn.benchmark     = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark     = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Dataset config
    # ------------------------------------------------------------------
    dataset_config = {
        'Synapse': {
            'Dataset':     Synapse_dataset,
            'volume_path': './data/Synapse/test_vol_h5',
            'list_dir':    './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing':   1,
        },
    }
    dataset_name     = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset     = dataset_config[dataset_name]['Dataset']
    args.list_dir    = dataset_config[dataset_name]['list_dir']
    args.z_spacing   = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    args.exp         = 'TU_' + dataset_name + str(args.img_size)

    # ------------------------------------------------------------------
    # Resolve snapshot -> a single .pth file
    #
    #  Case A: --snapshot_path ends with .pth  -> use it directly
    #  Case B: --snapshot_path is a folder     -> search inside it
    #  Case C: --snapshot_path is empty        -> auto-build folder, then search
    # ------------------------------------------------------------------
    sp = args.snapshot_path

    if sp.endswith('.pth'):
        # Case A - direct file
        if not os.path.isfile(sp):
            raise FileNotFoundError(f"Checkpoint file not found:\n  {sp}")
        snapshot = sp

    else:
        # Case B / C - resolve folder first
        if sp == '':
            # Auto-build folder path (original TransUNet logic)
            sp = "../model/{}/{}".format(args.exp, 'TU')
            sp = sp + '_pretrain' if args.is_pretrain else sp
            sp += '_' + args.vit_name
            sp  = sp + '_skip' + str(args.n_skip)
            if args.vit_patches_size != 16:
                sp = sp + '_vitpatch' + str(args.vit_patches_size)
            if args.max_epochs != 30:
                sp = sp + '_epo' + str(args.max_epochs)
            sp  = sp + '_bs' + str(args.batch_size)
            if args.base_lr != 0.01:
                sp = sp + '_lr' + str(args.base_lr)
            sp  = sp + '_' + str(args.img_size)
            if args.seed != 1234:
                sp = sp + '_s' + str(args.seed)

        if not os.path.isdir(sp):
            raise FileNotFoundError(
                f"Snapshot folder not found:\n  {sp}\n"
                "Pass --snapshot_path with the exact folder or .pth file."
            )

        snapshot = find_checkpoint(sp, args.max_epochs)
        if snapshot is None:
            raise FileNotFoundError(f"No checkpoint (.pth) found in:\n  {sp}")

    print(f"Loading checkpoint: {snapshot}")

    # ------------------------------------------------------------------
    # Build model and load weights
    # ------------------------------------------------------------------
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes    = args.num_classes
    config_vit.n_skip       = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )

    net = ViT_seg(config_vit, img_size=args.img_size,
                  num_classes=config_vit.n_classes).cuda()
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = os.path.basename(snapshot)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_folder = os.path.join('./test_log', 'test_log_' + args.exp)
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, snapshot_name.replace('.pth', '.txt'))
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"Checkpoint : {snapshot}")

    # ------------------------------------------------------------------
    # Optional NIfTI saving
    # ------------------------------------------------------------------
    if args.is_savenii:
        test_save_path = os.path.join(
            args.test_save_dir, args.exp, snapshot_name.replace('.pth', '')
        )
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)