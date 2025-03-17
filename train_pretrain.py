import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer
import deepspeed

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset
from eval_model import get_args as get_args4eval
from eval_model import main as main4eval

warnings.filterwarnings('ignore')


def Logger(content, ddp=None):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb,
                train_loader,
                iter_per_epoch,
                optimizer,
                ctx,
                model,
                scaler,
                ddp,
                lm_config,
                ):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if args.debug_step and step > args.debug_step:
            break
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        if args.use_deepspeed:
            model.backward(loss)
            model.step()
        else:
            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60), ddp)

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if step and step % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()
            
        if step and args.eval_interval and step % args.eval_interval == 0 and (not ddp or dist.get_rank() == 0):
            eval_args = get_args4eval()
            eval_args.max_seq_len = args.max_seq_len
            eval_args.model_mode = 0  # 0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型
            eval_args.use_deepspeed = args.use_deepspeed
            eval_args.dtype = args.dtype
            main4eval(eval_args)


def init_model(lm_config, ddp):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万', ddp)
    return model, tokenizer


def init_distributed_mode(ddp):
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    print(f'init_distributed_mode: ddp_local_rank={ddp_local_rank}, DEVICE={DEVICE}')


def main(args):
    print(f'args: {args}')

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    global ddp_local_rank, DEVICE
    ddp_local_rank, DEVICE = 0, "cuda:0"
    print('ddp:', ddp, os.environ.get("RANK", -1))

    if ddp:
        print(f'before init_distributed_mode: ddp_local_rank={ddp_local_rank}, DEVICE={DEVICE}')
        init_distributed_mode(ddp)
        print(f'after  init_distributed_mode: ddp_local_rank={ddp_local_rank}, DEVICE={DEVICE}')
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config, ddp)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    base_optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.use_deepspeed:
        ds_config = {
            "zero_optimization": {"stage": args.zero_stage},
            "train_batch_size": args.batch_size * args.accumulation_steps,
            "gradient_accumulation_steps": args.accumulation_steps,
            # Removed fixed fp16 setting to allow conditional configuration
            "pipeline": {
                "stages": 1,
                "async_io": True,
                "overlap_comm": True,
                "pipeline_micro_batches": 4  # set number of micro-batches
            }
        }
        if args.dtype == 'bfloat16':
            ds_config["bf16"] = {"enabled": True}
        elif args.dtype == 'float16':
            ds_config["fp16"] = {"enabled": True}
        model, base_optimizer, _, _ = deepspeed.initialize(args=args,
                                                           model=model,
                                                           optimizer=base_optimizer,
                                                           config=ds_config)

    elif ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        print('DistributedDataParallel:', ddp_local_rank)
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb, 
                train_loader,
                iter_per_epoch,
                base_optimizer,
                ctx,
                model,
                scaler,
                ddp,
                lm_config)
        eval_args = get_args4eval()
        eval_args.max_seq_len = args.max_seq_len
        eval_args.model_mode = 0  # 0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型
        eval_args.use_deepspeed = args.use_deepspeed
        eval_args.dtype = args.dtype
        main4eval(eval_args)
    del model, tokenizer


def get_args():
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)  # batch_size=64 50min pretrain/epoch
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")  # bfloat16  float16
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--debug_step', default=0, type=int)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    parser.add_argument('--use_deepspeed', default=False, type=bool, help="Enable DeepSpeed pipeline")
    parser.add_argument('--zero_stage', type=int, default=0, help="ZeRO optimization stage")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f'train_pretrain args unknow: {unknown}')
    return args
    

# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    args = get_args()
    main(args)
