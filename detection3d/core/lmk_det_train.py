import importlib
import numpy as np
import os
import shutil
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from detection3d.utils.image_tools import save_intermediate_results
from detection3d.loss.focal_loss import FocalLoss
from detection3d.loss.SigmoidFocalLoss import SigmoidFocalLossMultiChannel
from detection3d.utils.file_io import load_config, setup_logger, get_run_dir
from detection3d.utils.model_io import load_checkpoint, save_landmark_detection_checkpoint
from detection3d.dataset.dataloader import get_landmark_detection_dataloader


# ------------------------------------------------------------------------------
# ✅ EARLY STOPPING
# ------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4, start_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch

        self.best = float("inf")
        self.best_epoch = -1
        self.counter = 0

    def step(self, val_loss, epoch):
        """Returns True if should stop."""
        if epoch < self.start_epoch:
            return False

        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ------------------------------------------------------------------------------
# ✅ LOSS HANDLER (both mask styles)
# ------------------------------------------------------------------------------
def compute_landmark_mask_loss(outputs, landmark_masks, loss_function):
    """
    Supports both:
      Style 1: Single-channel multi-class mask (target shape [B,1,D,H,W])
      Style 2: Multi-channel ternary mask      (target shape [B,C,D,H,W])

    Mask values:
      Style 1: {-1,0,1,2,...}
      Style 2: {-1,0,1} in every channel

    outputs: logits [B,C,D,H,W]
    """

    assert outputs.shape[0] == landmark_masks.shape[0], "Batch size mismatch"

    # -------------------------------------------------------------------------
    # ✅ Style 1: single-channel multi-class
    # -------------------------------------------------------------------------
    if landmark_masks.shape[1] == 1:
        outputs = outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, outputs.shape[1])
        landmark_masks = landmark_masks.permute(0, 2, 3, 4, 1).contiguous().view(-1)

        # ignore (-1) externally
        valid_idx = torch.nonzero(landmark_masks >= 0, as_tuple=False).view(-1)
        outputs = torch.index_select(outputs, 0, valid_idx)
        landmark_masks = torch.index_select(landmark_masks, 0, valid_idx)

        loss = loss_function(outputs, landmark_masks)
        return loss

    # -------------------------------------------------------------------------
    # ✅ Style 2: multi-channel ternary (ignore differs per channel)
    # -------------------------------------------------------------------------
    else:
        # external ignore mask
        valid_mask = (landmark_masks >= 0).float()  # [B,C,D,H,W]

        # convert target {-1,0,1} -> {0,0,1}
        target = (landmark_masks > 0).float()

        loss = loss_function(outputs, target, valid_mask=valid_mask)
        return loss


# ------------------------------------------------------------------------------
# ✅ MASK STATS (tensorboard debug)
# ------------------------------------------------------------------------------
def compute_mask_stats(landmark_masks):
    """
    Works for:
      [B,1,D,H,W] multi-class (-1,0,1..K)
      [B,C,D,H,W] ternary     (-1,0,1)

    Returns:
        pos_fraction: fraction of positive voxels among valid voxels
        ignore_fraction: fraction of ignore voxels among all voxels
        zero_pos_rate: fraction of samples with zero positives
    """
    B = landmark_masks.shape[0]
    mask_flat = landmark_masks.view(B, -1)

    valid = mask_flat >= 0
    ignore = mask_flat < 0
    pos = mask_flat > 0

    valid_count = valid.sum(dim=1).float()
    pos_count = pos.sum(dim=1).float()
    ignore_count = ignore.sum(dim=1).float()

    pos_fraction = (pos_count / (valid_count + 1e-8)).mean().item()
    ignore_fraction = (ignore_count / mask_flat.shape[1]).mean().item()
    zero_pos_rate = (pos_count == 0).float().mean().item()

    return pos_fraction, ignore_fraction, zero_pos_rate


# ------------------------------------------------------------------------------
# ✅ TRAIN STEP (AMP + grad accumulation)
# ------------------------------------------------------------------------------
def train_step(cfg, net, crops, landmark_masks, frames, filenames,
               loss_func, optimizer, scaler, writer,
               batch_idx, accum_steps=1):

    net.train()

    if cfg.general.num_gpus > 0:
        crops = crops.cuda(non_blocking=True)
        landmark_masks = landmark_masks.cuda(non_blocking=True)

    # zero gradients only at start of accumulation window
    if batch_idx % accum_steps == 0:
        optimizer.zero_grad(set_to_none=True)

    with autocast(device_type="cuda", enabled=cfg.train.use_amp):
        outputs = net(crops)
        loss = compute_landmark_mask_loss(outputs, landmark_masks, loss_func)
        loss_scaled = loss / accum_steps

    if cfg.train.use_amp:
        scaler.scale(loss_scaled).backward()
    else:
        loss_scaled.backward()

    did_step = False

    if (batch_idx + 1) % accum_steps == 0:
        if cfg.train.use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        did_step = True

    # stats
    pos_frac, ignore_frac, zero_pos_rate = compute_mask_stats(landmark_masks)
    writer.add_scalar("MaskStats/pos_fraction", pos_frac, batch_idx)
    writer.add_scalar("MaskStats/ignore_fraction", ignore_frac, batch_idx)
    writer.add_scalar("MaskStats/zero_pos_rate", zero_pos_rate, batch_idx)

    # optional save debug
    if cfg.debug.save_inputs:
        batch_size = crops.size(0)
        save_path = os.path.join(cfg.general.save_dir, f"batch_{batch_idx}")
        save_intermediate_results(
            list(range(batch_size)), crops, landmark_masks, outputs,
            frames, filenames, save_path
        )

    return loss.detach(), did_step


# ------------------------------------------------------------------------------
# ✅ VALIDATION
# ------------------------------------------------------------------------------
def val_step(cfg, network, val_data_loader, loss_function):
    network.eval()
    val_loss_epoch = 0.0
    total_samples = 0

    with torch.no_grad():
        for _, (crops, landmark_masks, _, _, _) in enumerate(val_data_loader):
            batch_size = crops.size(0)

            if cfg.general.num_gpus > 0:
                crops = crops.cuda(non_blocking=True)
                landmark_masks = landmark_masks.cuda(non_blocking=True)

            with autocast(device_type="cuda", enabled=cfg.train.use_amp):
                outputs = network(crops)
                val_loss = compute_landmark_mask_loss(outputs, landmark_masks, loss_function)

            val_loss_epoch += val_loss.item()
            total_samples += batch_size

    return val_loss_epoch / max(total_samples, 1)


# ------------------------------------------------------------------------------
# ✅ LOG SETUP
# ------------------------------------------------------------------------------
def log_training_setup(cfg, logger):
    logger.info("========== Training Configuration ==========")
    logger.info(">>> Training Parameters")
    logger.info(f"AMP enabled           : {cfg.train.use_amp}")
    logger.info(f"Resume epoch          : {cfg.general.resume_epoch}")
    logger.info(f"Batch size            : {cfg.train.batch_size}")
    logger.info(f"Grad accum steps      : {cfg.train.grad_accum_steps}")
    logger.info(f"Learning rate         : {cfg.train.lr}")
    logger.info(f"Crop size             : {cfg.dataset.crop_size}")
    logger.info(f"Loss function         : {cfg.landmark_loss.name}")
    logger.info(f"Save directory        : {cfg.general.save_dir}")
    logger.info("============================================")


# ------------------------------------------------------------------------------
# ✅ TRAIN DRIVER
# ------------------------------------------------------------------------------
def train(config_file):
    assert os.path.isfile(config_file), f"Config not found: {config_file}"

    cfg = load_config(config_file)
    mode = "current" if cfg.general.resume_epoch >= 0 else "next"
    cfg.general.save_dir = get_run_dir(cfg.general.save_dir, mode)
    os.makedirs(cfg.general.save_dir, exist_ok=True)

    scaler = GradScaler() if cfg.train.use_amp else None

    np.random.seed(cfg.debug.seed)
    torch.manual_seed(cfg.debug.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.debug.seed)

    # dataloaders
    train_loader, num_modality, num_landmark_classes, num_train_cases = get_landmark_detection_dataloader(cfg, "train")
    val_loader, _, _, num_val_cases = get_landmark_detection_dataloader(cfg, "val")

    # network
    net_module = importlib.import_module("detection3d.network." + cfg.net.name)
    net = net_module.Net(num_modality, num_landmark_classes)
    max_stride = net.max_stride()
    net_module.parameters_kaiming_init(net)

    opt = optim.AdamW(net.parameters(), lr=cfg.train.lr,
                      betas=cfg.train.betas, weight_decay=cfg.train.weight_decay)

    if cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(cfg.dataset.crop_size) % max_stride == 0), "Crop size not divisible by stride"

    # resume
    if cfg.general.resume_epoch >= 0:
        start_epoch, start_batch_idx = load_checkpoint(cfg.general.resume_epoch, net, opt, scaler, cfg.general.save_dir)
    else:
        start_epoch, start_batch_idx = 0, 0

    # loss
    if cfg.landmark_loss.name == "Focal":
        loss_func = FocalLoss(
            class_num=num_landmark_classes,
            alpha=cfg.landmark_loss.focal_obj_alpha,
            gamma=cfg.landmark_loss.focal_gamma,
            use_gpu=cfg.general.num_gpus > 0,
        )
    elif cfg.landmark_loss.name == "SigmoidFocal":
        loss_func = SigmoidFocalLossMultiChannel(
            class_num=num_landmark_classes,
            alpha=cfg.landmark_loss.focal_obj_alpha,
            gamma=cfg.landmark_loss.focal_gamma,
        )
    else:
        raise ValueError(f"Unknown loss: {cfg.landmark_loss.name}")

    # writer + logger
    writer = SummaryWriter(os.path.join(cfg.general.save_dir, "tensorboard"))
    shutil.copy(config_file, os.path.join(cfg.general.save_dir, os.path.basename(config_file)))

    logger = setup_logger(os.path.join(cfg.general.save_dir, "train_log.txt"), "lmk_det3d")
    log_training_setup(cfg, logger)

    # CSV logging
    csv_path = os.path.join(cfg.general.save_dir, "loss_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["epoch", "train_loss", "val_loss", "best_val_loss", "best_epoch"])

    # early stopper
    early_stopper = EarlyStopping(
        patience=cfg.train.early_stop_patience,
        min_delta=cfg.train.early_stop_min_delta,
        start_epoch=cfg.train.early_stop_start_epoch,
    )

    # tracking
    best_val_loss = float("inf")
    best_epoch = -1

    train_loss_epoch = 0.0
    train_batch_size = 0
    prev_epoch_idx = start_epoch
    optimizer_step_idx = 0

    loader_iter = tqdm(enumerate(train_loader, start=start_batch_idx), total=len(train_loader), desc="Training", unit="batch")

    logger.info(f"Training started with {num_train_cases} train cases, {num_val_cases} val cases")

    batch_idx = start_batch_idx

    for _, (crops, landmark_masks, _, frames, filenames) in loader_iter:

        begin_t = time.time()
        batch_size = crops.size(0)
        train_batch_size += batch_size

        loss, did_step = train_step(
            cfg, net, crops, landmark_masks, frames, filenames,
            loss_func, opt, scaler, writer,
            batch_idx=batch_idx,
            accum_steps=cfg.train.grad_accum_steps,
        )

        train_loss_epoch += loss.item()
        writer.add_scalar("Loss_batch/TrainLoss", loss.item(), batch_idx)

        if did_step:
            optimizer_step_idx += 1
            writer.add_scalar("Optimizer/step_idx", optimizer_step_idx, batch_idx)

        # epoch estimate
        epoch_idx = batch_idx * cfg.train.batch_size // num_train_cases
        sample_duration = (time.time() - begin_t) / batch_size

        # epoch finished
        if epoch_idx > prev_epoch_idx:
            avg_train_loss = train_loss_epoch / max(train_batch_size, 1)
            avg_val_loss = val_step(cfg, net, val_loader, loss_func)

            # tensorboard
            writer.add_scalars("Loss_epoch", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch_idx)

            # ✅ always save "current"
            save_landmark_detection_checkpoint(
                net, opt, scaler,
                epoch_idx, batch_idx,
                cfg, config_file,
                max_stride, num_modality,
                tag="current"
            )

            # ✅ update best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch_idx

                logger.info(f"[BEST] val improved -> {best_val_loss:.6f} (epoch {best_epoch})")

                save_landmark_detection_checkpoint(
                    net, opt, scaler,
                    epoch_idx, batch_idx,
                    cfg, config_file,
                    max_stride, num_modality,
                    tag="best"
                )

            # CSV logging
            with open(csv_path, "a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([epoch_idx, avg_train_loss, avg_val_loss, best_val_loss, best_epoch])

            # print log
            logger.info(
                f"epoch {epoch_idx} | train {avg_train_loss:.6f} | val {avg_val_loss:.6f} "
                f"| best {best_val_loss:.6f} @ {best_epoch} | time {sample_duration:.4f}s/vol"
            )

            # early stop
            should_stop = early_stopper.step(avg_val_loss, epoch_idx)
            writer.add_scalar("EarlyStop/best_val_loss", early_stopper.best, epoch_idx)
            writer.add_scalar("EarlyStop/patience_counter", early_stopper.counter, epoch_idx)

            # reset
            train_loss_epoch = 0.0
            train_batch_size = 0
            prev_epoch_idx = epoch_idx

            if should_stop:
                logger.info(
                    f"[EARLY STOP] No improvement for {early_stopper.patience} epochs. "
                    f"Best val {early_stopper.best:.6f} @ epoch {early_stopper.best_epoch}. Stop at epoch {epoch_idx}."
                )
                break

        batch_idx += 1

    # flush leftover gradients at end
    if (batch_idx % cfg.train.grad_accum_steps) != 0:
        logger.info("[INFO] Flushing remaining accumulated gradients at end...")
        if cfg.train.use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

    writer.close()
    logger.info(f"Training finished. Save directory: {cfg.general.save_dir}")
    logger.info(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
