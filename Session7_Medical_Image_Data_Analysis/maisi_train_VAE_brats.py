import os
import sys
import json
import yaml
import glob
import wandb
import signal
import argparse
import warnings
import numpy as np
from tqdm import trange

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.config import print_config
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import PatchDiscriminator
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.data import CacheDataset, DataLoader, DistributedSampler

from utils import count_parameters
from scripts.transforms import VAE_Transform
from scripts.utils import define_instance, dynamic_infer #, KL_loss
from scripts.utils_plot import find_label_center_loc, get_xyz_plot

warnings.filterwarnings("ignore")
if os.environ.get("RANK", "0") == "0":
    print_config()

def setup_ddp(): # DDP CHANGE: 분산 환경 초기화 함수
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp(): # DDP CHANGE: 분산 환경 정리 함수
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

SHUTDOWN_REQUESTED = False
def graceful_shutdown(signum, frame):
    global SHUTDOWN_REQUESTED
    print(f"\n[!] Received signal {signum}. Requesting graceful shutdown...", flush=True)
    SHUTDOWN_REQUESTED = True
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

def get_run_name(manual_name=None, default_prefix="manual"):
    job_id = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME")
    if manual_name:
        return manual_name
    elif job_id and job_name:
        return f"{job_name}_{job_id}"
    elif job_id:
        return f"slurm_{job_id}"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{default_prefix}_{timestamp}"

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi3d-rflow_brats.json")
    parser.add_argument("--train_config_path", type=str, default="/shared/s1/lab06/wonyoung/maisi/configs/config_maisi_vae_train_brats.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cpus_per_task", type=int, default=8, help="Number of CPUs allocated per task by Slurm.")
    args = parser.parse_args()
    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run_name to be specified.")
    args.run_name = get_run_name(args.run_name)
    config_dict = json.load(open(args.model_config_path, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)
    config_train_dict = json.load(open(args.train_config_path, "r"))
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
    for k, v in config_train_dict["custom_config"].items():
        setattr(args, k, v)
    return args

def resume_from_latest(autoencoder,
                       discriminator,
                       optimizer_g, 
                       optimizer_d,
                       scheduler_g,
                       scheduler_d,
                       scaler_g,
                       scaler_d,
                       output_dir, device):
    if not os.path.exists(output_dir):
        print("[Resume] No checkpoint directory found. Starting fresh.")
        return 0, float("inf"), 0

    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, d))
        and os.path.exists(os.path.join(output_dir, d, "model.pt"))
    ]
    if len(ckpts) == 0:
        print("[Resume] No checkpoint found in directory. Starting fresh.")
        return 0, float("inf"), 0

    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    latest_ckpt_dir = ckpts[-1]
    print(f"[Resume] Resuming from checkpoint: {latest_ckpt_dir}")
    
    path = os.path.join(output_dir, latest_ckpt_dir, "model.pt")
    ckpt = torch.load(path, map_location=device)
    
    autoencoder.load_state_dict(ckpt["autoencoder"])
    discriminator.load_state_dict(ckpt["discriminator"])
    optimizer_g.load_state_dict(ckpt["optimizer_g"]) ### comment this line if s2 first submit
    optimizer_d.load_state_dict(ckpt["optimizer_d"]) ### comment this line if s2 first submit
    if "scheduler_g" in ckpt: scheduler_g.load_state_dict(ckpt["scheduler_g"]) ### comment this line if s2 first submit
    if "scheduler_d" in ckpt: scheduler_d.load_state_dict(ckpt["scheduler_d"]) ### comment this line if s2 first submit
    if "scaler_g" in ckpt and scaler_g is not None: scaler_g.load_state_dict(ckpt["scaler_g"]) ### comment this line if s2 first submit
    if "scaler_d" in ckpt and scaler_d is not None: scaler_d.load_state_dict(ckpt["scaler_d"]) ### comment this line if s2 first submit

    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf")), ckpt.get("epoch", 0)

def prepare_image_for_logging(image_tensor, center_loc):
    """3D 텐서를 wandb에 로깅할 수 있는 2D 이미지로 변환합니다."""
    image_tensor_cpu = image_tensor.cpu()
    vis_img_np = get_xyz_plot(image_tensor_cpu, center_loc, mask_bool=False)
    min_val, max_val = vis_img_np.min(), vis_img_np.max()
    if max_val - min_val > 1e-6:
        vis_img_np = (vis_img_np - min_val) / (max_val - min_val)
    else:
        vis_img_np = np.zeros_like(vis_img_np)
    vis_img_uint8 = (vis_img_np * 255).astype(np.uint8)
    return wandb.Image(vis_img_uint8)

def reduce_mean_scalar(x):
    t = x.detach().reshape(1).to(torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()

def main():
    setup_ddp() # DDP CHANGE: 분산 설정 초기화
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print("="*50)
        print(f"✅ Training started with a total of {world_size} GPUs across all nodes.")
        print("="*50)

    args = load_config()
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = torch.float16 if args.weight_dtype == "fp16" else torch.float32

    if rank == 0:
        print(f"[Opt] Using gradient accumulation with {args.gradient_accumulation_steps} steps.")

    set_determinism(seed=args.seed + rank) # OPTIMIZATION: 각 프로세스에 다른 시드 줘서 데이터 증강 다양성 확보
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True # OPTIMIZATION: 입력 크기가 일정할 때 가장 효과적이므로 활성화
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    if rank == 0:
        print("[Config] Loaded hyperparameters:")
        print(yaml.dump(vars(args), sort_keys=False))
        if args.report_to:
            wandb.init(project="MAISI_EX_DDP", config=args, name=args.run_name)
        output_dir = os.path.join(args.output_dir, args.run_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    # BraTS Data
    train_files_list = [{"image": f} for f in sorted(glob.glob(os.path.join(args.train_label_dir, "*", "*.nii.gz"))) if "seg" not in f]
    valid_filepaths = [f for f in sorted(glob.glob(os.path.join(args.valid_label_dir, "*", "*.nii.gz"))) if "seg" not in f]
    val_files_list = [{"image": f} for f in valid_filepaths[:args.num_valid]]
    
    def add_assigned_class_to_datalist(datalist, classname): # 데이터에 'class' 키를 추가하는 함수
        for item in datalist:
            item["class"] = classname
        return datalist

    train_files = add_assigned_class_to_datalist(train_files_list, "mri")
    val_files = add_assigned_class_to_datalist(val_files_list, "mri")

    train_transform = VAE_Transform(
        is_train=True, 
        random_aug=args.random_aug, 
        k=4, 
        patch_size=args.patch_size, 
        output_dtype=weight_dtype, 
        spacing_type=args.spacing_type, 
        image_keys=["image"]
    )
    val_transform = VAE_Transform(
        is_train=False, 
        random_aug=False, 
        k=4, 
        output_dtype=weight_dtype, 
        image_keys=["image"]
    )

    # Build dataloader
    if rank == 0: print(f"Total number of training data is {len(train_files)}.")
    workers_per_gpu = args.cpus_per_task // world_size # 각 프로세스가 사용할 수 있는 CPU 코어 수에 맞춰 num_workers 설정
    train_dataset = CacheDataset(data=train_files, transform=train_transform, cache_rate=args.cache, num_workers=workers_per_gpu)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    dataloader_train = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=workers_per_gpu, 
        sampler=train_sampler, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    steps_per_epoch = len(dataloader_train)
    
    if rank == 0: print(f"Total number of validation data is {len(val_files)}.")
    val_total = len(val_files)
    per_rank = (val_total + world_size - 1) // world_size  # ceil
    start = rank * per_rank
    end = min(val_total, start + per_rank)
    val_files_shard = val_files[start:end]
    dataset_val = CacheDataset(data=val_files_shard, transform=val_transform, cache_rate=0.0, num_workers=workers_per_gpu)
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=args.val_batch_size, 
        num_workers=workers_per_gpu, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    if rank == 0:
        print("### Train Transform ###")
        for i, t in enumerate(train_transform.transform_dict["mri"].transforms):
            print(f"[{i}] {t}")
        print("\n### Validation Transform ###")
        for i, t in enumerate(val_transform.transform_dict["mri"].transforms):
            print(f"[{i}] {t}")

    # https://github.com/Project-MONAI/MONAI/blob/e267705385d00ef0071cf51b087345d720af9102/monai/networks/nets/autoencoderkl.py#L472
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # https://github.com/Project-MONAI/MONAI/blob/e267705385d00ef0071cf51b087345d720af9102/monai/networks/nets/patchgan_discriminator.py#L116
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)

    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    total_opt_steps = (args.max_train_steps - args.pretrained_steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps ### check pretrained_steps when s2
    scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=total_opt_steps)
    scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=total_opt_steps)
    scaler_g, scaler_d = (GradScaler(), GradScaler()) if args.amp else (None, None)

    start_step, best_val_loss, start_epoch = 0, float("inf"), 0
    if args.resume:
        output_dir_for_resume = os.path.join(args.output_dir, args.run_name)
        start_step, best_val_loss, start_epoch = resume_from_latest(
            autoencoder, discriminator, optimizer_g, optimizer_d,
            scheduler_g, scheduler_d, scaler_g, scaler_d, output_dir_for_resume, device)
    dist.barrier(device_ids=[local_rank])

    # OPTIMIZATION: torch.compile로 모델 컴파일하여 연산 가속화 (PyTorch 2.0+ 필수)
    if rank == 0: print("Compiling models with torch.compile()...")
    autoencoder = torch.compile(autoencoder)
    discriminator = torch.compile(discriminator)
    dist.barrier(device_ids=[local_rank])
    autoencoder = DDP(autoencoder, device_ids=[local_rank], find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)

    sync_vec = torch.tensor([start_step, float(best_val_loss)], device=device, dtype=torch.float32)
    dist.broadcast(sync_vec, src=0)
    start_step, best_val_loss = int(sync_vec[0].item()), float(sync_vec[1].item())

    # Training config
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    
    if rank == 0:
        print("### start_epoch:", start_epoch)
        param_counts = count_parameters(autoencoder.module)
        print(f"### autoencoder's Trainable parameters: {param_counts['trainable']:,}") # 20,944,897
        param_counts = count_parameters(discriminator.module)
        print(f"### discriminator's Trainable parameters: {param_counts['trainable']:,}") # 2,770,977

    def infinite_loader(loader, sampler, start_epoch=0):
        epoch = start_epoch
        while True:
            sampler.set_epoch(epoch) # DDP CHANGE: 매 epoch마다 샘플러 시드 변경
            for batch in loader:
                yield batch
            epoch += 1
            
    train_iter = infinite_loader(dataloader_train, train_sampler, start_epoch)
    progress_bar = trange(start_step, args.max_train_steps + 1,
                          desc=f"Training on Rank {rank}",
                          initial=start_step, total=args.max_train_steps + 1,
                          disable=(rank != 0))
        
    for step in progress_bar:
        autoencoder.train()
        discriminator.train()
        batch = next(train_iter)
        # OPTIMIZATION: non_blocking=True로 데이터 전송과 연산 오버랩 시도
        images = batch["image"].to(device, non_blocking=True).contiguous()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            reconstruction, z_mu, z_sigma = autoencoder(images)

        z_mu_f = z_mu.float()
        z_sigma_f = torch.clamp(z_sigma.float(), min=1e-8)
        logvar = 2.0 * torch.log(z_sigma_f)                # σ -> log σ²
        logvar = torch.clamp(logvar, min=-30.0, max=10.0)  # 상한 10 정도로 낮추는 게 안전
        kl = 0.5 * (torch.exp(logvar) + z_mu_f**2 - 1.0 - logvar)
        kl_loss = kl.mean()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            losses = {
                "recons_loss": intensity_loss(reconstruction, images), # voxel 직접 비교
                "kl_loss": kl_loss, # 압축된 데이터가 정규분포를 따르도록 강제 (latent 다듬는 역할 -> loss 적절하게 유지되어야 함; 너무 크면 생성 fail, 너무 작으면 생성 다 똑같아짐)
                "p_loss": loss_perceptual(reconstruction.float(), images.float()), # 시각적 차이를 반영
            }
            # Gen이 Disc. 얼마나 잘 속이는가?
            # loss 낮을수록 좋지만 Gen이 너무 잘해서 0이면 Disc.가 제 역할 못한다는 뜻
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + \
                     args.perceptual_weight * losses["p_loss"] + args.adv_weight * generator_loss
            loss_g = loss_g / args.gradient_accumulation_steps
            
        if args.amp:
            scaler_g.scale(loss_g).backward()
        else:
            loss_g.backward()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            # 가짜 이미지 보여줬을 때 가짜라고 얼마나 잘 맞혔는가? 
            # loss 낮을수록 좋지만 Disc.가 너무 잘해서 0이면 Gen이 배울 게 없어짐
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            # 진짜 이미지 보여줬을 때 진짜라고 얼마나 잘 맞혔는가? 
            # loss 낮을수록 좋지만 Disc.가 너무 잘해서 0이면 Gen이 배울 게 없어짐
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d = loss_d / args.gradient_accumulation_steps

        if args.amp:
            scaler_d.scale(loss_d).backward()
        else:
            loss_d.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Generator 업데이트
            if args.amp:
                scaler_g.unscale_(optimizer_g)
                clip_grad_norm_(autoencoder.parameters(), 1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                clip_grad_norm_(autoencoder.parameters(), 1.0)
                optimizer_g.step()
            scheduler_g.step()
            optimizer_g.zero_grad(set_to_none=True)

            # Discriminator 업데이트
            if args.amp:
                scaler_d.unscale_(optimizer_d)
                clip_grad_norm_(discriminator.parameters(), 1.0)
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_d.step()
            scheduler_d.step()
            optimizer_d.zero_grad(set_to_none=True)

        # DDP CHANGE: 모든 GPU의 loss를 평균내어 로그 기록
        lg = reduce_mean_scalar(loss_g) * args.gradient_accumulation_steps
        ld = reduce_mean_scalar(loss_d) * args.gradient_accumulation_steps
        avg_recons_loss = reduce_mean_scalar(losses["recons_loss"]) * args.gradient_accumulation_steps
        avg_kl_loss = reduce_mean_scalar(losses["kl_loss"]) * args.gradient_accumulation_steps
        avg_p_loss = reduce_mean_scalar(losses["p_loss"]) * args.gradient_accumulation_steps
        avg_gen_loss = reduce_mean_scalar(generator_loss) * args.gradient_accumulation_steps
        avg_dfake_loss = reduce_mean_scalar(loss_d_fake) * args.gradient_accumulation_steps
        avg_dreal_loss = reduce_mean_scalar(loss_d_real) * args.gradient_accumulation_steps

        if rank == 0:
            progress_bar.set_postfix({'Total_g_loss': f"{lg:.4f}", 'Total_d_loss': f"{ld:.4f}"})
            if args.report_to and step % 100 == 0: # 로그 기록 빈도 조절
                log_data = {
                    "train/learning_rate": scheduler_g.get_last_lr()[0],
                    "train/loss_g_total": lg,
                    "train/loss_d_total": ld,
                    "train/generator/recons_loss": avg_recons_loss,
                    "train/generator/kl_loss": avg_kl_loss,
                    "train/generator/p_loss": avg_p_loss,
                    "train/discriminator/adv_g_loss": avg_gen_loss,
                    "train/discriminator/d_fake_loss": avg_dfake_loss,
                    "train/discriminator/d_real_loss": avg_dreal_loss,
                }
                wandb.log(log_data, step=step)

        did_validate = False
        if (step % args.validation_steps == 0 or step == args.max_train_steps) and step > start_step:
            did_validate = True
            autoencoder.eval()
            val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
            num_val_batches_local = 0
            val_inferer = SlidingWindowInferer(
                roi_size=args.val_sliding_window_patch_size, 
                sw_batch_size=1, 
                overlap=0.5, 
                device=device, #torch.device("cpu"),
                sw_device=device
            )            
            with torch.no_grad():
                for val_batch in dataloader_val:
                    val_images = val_batch["image"].to(device)
                    # DDP CHANGE: .module을 통해 원본 모델로 추론
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                        reconstruction, z_mu_val, z_sigma_val = dynamic_infer(val_inferer, autoencoder.module, val_images)

                    z_mu_val_f = z_mu_val.float()
                    z_sigma_val_f = torch.clamp(z_sigma_val.float(), min=1e-8)
                    logvar_val = 2.0 * torch.log(z_sigma_val_f)                # σ -> log σ²
                    logvar_val = torch.clamp(logvar_val, min=-30.0, max=10.0)  # 상한 10 정도로 낮추는 게 안전
                    kl_val = 0.5 * (torch.exp(logvar_val) + z_mu_val_f**2 - 1.0 - logvar_val)
                    kl_loss_val = kl_val.mean().item()
                    
                    reconstruction = reconstruction.to(device)
                    val_images = val_images.to(device)
                    val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, val_images).item()
                    val_epoch_losses["kl_loss"] += kl_loss_val
                    val_epoch_losses["p_loss"] += loss_perceptual(reconstruction.float(), val_images.float()).item()
                    num_val_batches_local += 1

            val_metrics = torch.tensor(
                [val_epoch_losses["recons_loss"], val_epoch_losses["kl_loss"], val_epoch_losses["p_loss"], num_val_batches_local],
                device=device
            )
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
            total_batches = val_metrics[3].item()
            avg_recon_loss = val_metrics[0].item() / total_batches if total_batches > 0 else 0
            avg_kl_loss = val_metrics[1].item() / total_batches if total_batches > 0 else 0
            avg_p_loss = val_metrics[2].item() / total_batches if total_batches > 0 else 0
            final_val_losses = {"recons_loss": avg_recon_loss, "kl_loss": avg_kl_loss, "p_loss": avg_p_loss}
            val_loss_g = final_val_losses["recons_loss"] + \
                         args.kl_weight * final_val_losses["kl_loss"] + \
                         args.perceptual_weight * final_val_losses["p_loss"]

            if rank == 0:                
                print(f"\nStep {step} Total Val Loss (Avg across all ranks): {val_loss_g:.4f}, Details: {final_val_losses}")
                if args.report_to:
                    log_data = {
                        "valid/total_loss": val_loss_g,
                        "valid/recon_loss": final_val_losses["recons_loss"],
                        "valid/kl_loss": final_val_losses["kl_loss"],
                        "valid/p_loss": final_val_losses["p_loss"],
                    }
                    if num_val_batches_local > 0: # scale_factor와 이미지는 마지막 배치의 결과만 참고용으로 로깅
                        std = z_mu_val.detach().float().flatten().std().clamp(min=1e-8)
                        log_data["valid/scale_factor"] = (1.0 / std).item()
                        center_loc = find_label_center_loc(val_images[0, 0, ...])
                        log_data["valid/original_image"] = prepare_image_for_logging(val_images[0], center_loc)
                        log_data["valid/reconstructed_image"] = prepare_image_for_logging(reconstruction[0], center_loc)
                    wandb.log(log_data, step=step)
                
                if val_loss_g < best_val_loss:
                    torch.cuda.synchronize(device)
                    autoencoder_state_dict = autoencoder.module._orig_mod.state_dict()
                    discriminator_state_dict = discriminator.module._orig_mod.state_dict()
                    current_epoch = step // steps_per_epoch
                    best_val_loss = float(val_loss_g)
                    state = {
                        "autoencoder": autoencoder_state_dict,
                        "discriminator": discriminator_state_dict,
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict(),
                        "scheduler_d": scheduler_d.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "epoch": current_epoch
                    }
                    if args.amp:
                        state["scaler_g"] = scaler_g.state_dict()
                        state["scaler_d"] = scaler_d.state_dict()
                    best_dir = os.path.join(output_dir, "best-checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    atomic_save(state, os.path.join(best_dir, "model.pt"))
                    print(f"[best] updated at step {step}: {best_val_loss:.6f}", flush=True)
                else:
                    print(f"[not best] not updated at step {step}: {val_loss_g:.6f}", flush=True)

            _best = torch.tensor([best_val_loss], device=device, dtype=torch.float32)
            dist.broadcast(_best, src=0)
            best_val_loss = float(_best.item())

        # --- 체크포인트 저장 및 Validation ---
        is_time_to_save = (step % args.checkpointing_steps == 0 and step > start_step)
        if (is_time_to_save or SHUTDOWN_REQUESTED) and rank == 0:
            torch.cuda.synchronize(device)
            autoencoder_state_dict = autoencoder.module._orig_mod.state_dict()
            discriminator_state_dict = discriminator.module._orig_mod.state_dict()
            current_epoch = step // steps_per_epoch
            state = {
                "autoencoder": autoencoder_state_dict,
                "discriminator": discriminator_state_dict,
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "step": step,
                "best_val_loss": float(best_val_loss),
                "epoch": current_epoch
            }
            if args.amp:
                state["scaler_g"] = scaler_g.state_dict()
                state["scaler_d"] = scaler_d.state_dict()
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            atomic_save(state, os.path.join(ckpt_dir, "model.pt"))
            print(f"\nSaved checkpoint to {ckpt_dir}", flush=True)

        shutdown_tensor = torch.tensor([1 if (rank == 0 and SHUTDOWN_REQUESTED) else 0], device=device)
        dist.broadcast(shutdown_tensor, src=0)
        if shutdown_tensor.item() == 1:
            if rank == 0:
                print("Shutdown signal received and synced across all ranks. Exiting training loop gracefully.")
            break
        if did_validate:
            dist.barrier(device_ids=[local_rank])

    if SHUTDOWN_REQUESTED:
        print("Graceful shutdown initiated, exiting with code 1 to trigger requeue.")
        sys.exit(1)
    
    dist.barrier(device_ids=[local_rank]) # DDP CHANGE: 모든 프로세스가 끝날 때까지 대기 후 정리
    cleanup_ddp()

if __name__ == '__main__':
    main()
