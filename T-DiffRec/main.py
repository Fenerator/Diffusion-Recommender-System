"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy

import random

# NEW ====================
import wandb
import sys, os

# END NEW ================


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="yelp_clean", help="choose the dataset"
)
parser.add_argument(
    "--data_path", type=str, default="../datasets/", help="load data path"
)
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=400)
parser.add_argument("--epochs", type=int, default=1000, help="upper epoch limit")
parser.add_argument("--topN", type=str, default="[10, 20, 50, 100]")
parser.add_argument("--tst_w_val", action="store_true", help="test with validation")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
parser.add_argument(
    "--save_path", type=str, default="./saved_models/", help="save model path"
)
parser.add_argument("--log_name", type=str, default="log", help="the log name")
parser.add_argument("--round", type=int, default=1, help="record the experiment")

parser.add_argument(
    "--w_min", type=float, default=0.1, help="the minimum weight for interactions"
)
parser.add_argument(
    "--w_max", type=float, default=1.0, help="the maximum weight for interactions"
)

# params for the model
parser.add_argument("--time_type", type=str, default="cat", help="cat or add")
parser.add_argument("--dims", type=str, default="[1000]", help="the dims for the DNN")
parser.add_argument(
    "--norm", type=bool, default=False, help="Normalize the input or not"
)
parser.add_argument("--emb_size", type=int, default=10, help="timestep embedding size")

# params for diffusion
parser.add_argument(
    "--mean_type", type=str, default="x0", help="MeanType for diffusion: x0, eps"
)
# parser.add_argument("--steps", type=int, default=100, help="diffusion steps")
parser.add_argument(
    "--noise_schedule",
    type=str,
    default="linear-var",
    help="the schedule for noise generating",
)
parser.add_argument(
    "--noise_scale", type=float, default=0.1, help="noise scale for noise generating"
)
parser.add_argument(
    "--noise_min",
    type=float,
    default=0.0001,
    help="noise lower bound for noise generating",
)
parser.add_argument(
    "--noise_max",
    type=float,
    default=0.02,
    help="noise upper bound for noise generating",
)
parser.add_argument(
    "--sampling_noise", type=bool, default=False, help="sampling with noise or not"
)

parser.add_argument(
    "--reweight",
    type=bool,
    default=True,
    help="assign different weight to different timestep or not",
)

# NEW ====================
parser.add_argument("--num_workers", type=int, default=4, help="num of workers")

parser.add_argument(
    "--model_type", type=str, default="T-DiffRec", help="type DRS Model"
)

parser.add_argument(
    "--run_name", type=str, default="", help="run name extension for wandb"
)

parser.add_argument("--seed", type=int, default=1, help="random seed")

parser.add_argument(
    "--patience", type=int, default=20, help="patience for early stopping"
)

parser.add_argument(
    "--visualize_weights",
    action="store_true",
    help="visualize weights, only applicable if mean_type is x0_learnable",
)

parser.add_argument(
    "--workers", type=int, default=10, help="number of workers for m-phate"
)

parser.add_argument(
    "--attention_weighting",
    action="store_true",
    help="use the attention weighting feature",
)
# END NEW ====================

args = parser.parse_args()

# BEGIN NEW ====================
if args.dataset == "amazon-book_clean":
    args.stepsw = 10
    args.sampling_steps = 0
elif args.dataset == "yelp_clean":
    args.steps = 5
    args.sampling_steps = 0
elif args.dataset == "yelp_noisy":
    args.steps = 5
    args.sampling_steps = 0

elif args.dataset == "ml-1m_clean":
    args.steps = 5
    args.sampling_steps = 0

elif args.dataset == "ml-1m_noisy":
    args.steps = 5
    args.sampling_steps = 0
else:
    args.steps = 100
    args.sampling_steps = 0
# END NEW ====================

# NEW ====================
# import visualization function if needed
if args.mean_type == "x0_learnable" and args.visualize_weights:
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)

    from utils.visualize_weights import generate_m_phate, create_phate_visualization


if args.attention_weighting:
    args.reweight = False
    print(
        "Attention weighting is enabled, therefore standard reweighting was disabled."
    )
# END NEW ====================

print("args:", args)

random_seed = args.seed
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn

# NEW
# init wandb
wandb.init(
    name=f"{args.model_type}_{args.dataset}_{args.seed}_{args.run_name}",
    project="drs",
    notes="This is a test run",
    tags=[f"{args.model_type}", f"{args.dataset}"],
    entity="drs",
    config=args,
)
# END NEW


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print(
    "Starting time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
)

### DATA LOAD ###
train_path = f"{args.data_path}{args.dataset}/train_list.npy"
valid_path = f"{args.data_path}{args.dataset}/valid_list.npy"
test_path = f"{args.data_path}{args.dataset}/test_list.npy"

(
    train_data,
    train_data_ori,
    valid_y_data,
    test_y_data,
    n_user,
    n_item,
) = data_utils.data_load(train_path, valid_path, test_path, args.w_min, args.w_max)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    pin_memory=True,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=worker_init_fn,
)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

# BEGIN NEW ====================
if args.mean_type == "x0_learnable":
    train_data = train_data_ori
# END NEW ====================

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(
        torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A)
    )
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data_ori + valid_y_data

# NEW ====================
# load item embeddings
if args.attention_weighting:
    emb_path = f"{args.data_path}{args.dataset}/item_emb.npy"
    item_embeddings = torch.from_numpy(np.load(emb_path, allow_pickle=True))
else:
    item_embeddings = None
# END NEW ====================

print("data ready.")


### Build Gaussian Diffusion ###
if args.mean_type == "x0":
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == "eps":
    mean_type = gd.ModelMeanType.EPSILON
    # BEGIN NEW ====================
elif args.mean_type == "x0_learnable":
    mean_type = gd.ModelMeanType.LEARNABLE_PARAM
    # END NEW ====================
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(
    mean_type,
    args.noise_schedule,
    args.noise_scale,
    args.noise_min,
    args.noise_max,
    args.steps,
    device,
    attention_weighting=args.attention_weighting,
    item_embeddings=item_embeddings,
).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]
print("in_dims:", in_dims)
model = DNN(
    in_dims,
    out_dims,
    args.emb_size,
    time_type="cat",
    norm=args.norm,
    steps=args.steps,
    attention_weighting=args.attention_weighting,
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

wandb.watch(model)

print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)


# NEW ====================
def log_results(results, epoch, topN, mode="valid"):
    """Log results to wandb."""
    precisions, recalls, NDCGs, MRRs = results

    assert (
        len(precisions) == len(recalls) == len(NDCGs) == len(MRRs) == len(topN)
    ), f"Lengths not equal: {len(precisions)}, {len(recalls)}, {len(NDCGs)}, {len(MRRs)}, {len(topN)}"

    # log all metrics @ topN individually
    for i, k in enumerate(topN):
        wandb.log(
            {
                "Epoch": epoch,
                f"{mode} Precision@{k}": precisions[i],
                f"{mode} Recall@{k}": recalls[i],
                f"{mode} NDCG@{k}": NDCGs[i],
                f"{mode} MRR@{k}": MRRs[i],
            }
        )


# END NEW ====================


def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[
                e_idxlist[
                    batch_idx * args.batch_size : batch_idx * args.batch_size
                    + len(batch)
                ]
            ]
            batch = batch.to(device)
            prediction = diffusion.p_sample(
                model, batch, args.sampling_steps, args.sampling_noise
            )
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results


best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= args.patience:
        print("-" * 18)
        print("Exiting from training early")
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        """1 Batch is float tensor of bs x 2810 (item num)"""

        batch = batch.to(device)

        batch_count += 1
        optimizer.zero_grad()

        losses = diffusion.training_losses(
            model,
            batch,
            args.reweight,
        )
        loss = losses["loss"].mean()

        wandb.log({"batch_loss_train": loss})

        total_loss += loss
        loss.backward()
        optimizer.step()

    wandb.log({"epoch_loss_norm_train": total_loss / batch_count, "Epoch": epoch})

    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))

        log_results(valid_results, epoch, eval(args.topN), mode="valid")

        if args.tst_w_val:
            test_results = evaluate(
                test_twv_loader, test_y_data, mask_tv, eval(args.topN)
            )
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))

        log_results(test_results, epoch, eval(args.topN), mode="test")

        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall:  # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(
                model,
                "{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_wmin{}_wmax{}_{}.pth".format(
                    args.save_path,
                    args.dataset,
                    args.lr,
                    args.weight_decay,
                    args.batch_size,
                    args.dims,
                    args.emb_size,
                    args.mean_type,
                    args.steps,
                    args.noise_scale,
                    args.noise_min,
                    args.noise_max,
                    args.sampling_steps,
                    args.reweight,
                    args.w_min,
                    args.w_max,
                    args.log_name,
                ),
            )

    print(
        "Runing Epoch {:03d} ".format(epoch)
        + "train loss {:.4f}".format(total_loss)
        + " costs "
        + time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
    )
    print("---" * 18)


print("===" * 18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)
print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

log_results(best_results, best_epoch, eval(args.topN), mode="best_valid")
log_results(best_test_results, best_epoch, eval(args.topN), mode="best_test")


# NEW ====================
# Store learnable parameter
if args.mean_type == "x0_learnable":
    # stack the values into a single tensor
    params_per_batch = torch.stack(model.param_storage, dim=0)
    print(f"params_per_batch.shape: {params_per_batch.shape}")

    # store as npy
    params_per_batch = params_per_batch.detach().cpu().numpy()
    np.save(
        f"mPHATE/{args.model_type}_{args.dataset}_{args.seed}_{args.run_name}_params_per_batch.npy",
        params_per_batch,
    )
    print(
        f"Saved params_per_batch.npy to mPHATE/{args.model_type}_{args.dataset}_{args.seed}_{args.run_name}_params_per_batch.npy"
    )

if args.mean_type == "x0_learnable" and args.visualize_weights:
    # generate weight visualization
    print(
        f"params_per_batch shape: {params_per_batch.shape}"
    )  # n_time_steps, n_points, n_dim

    m_phate_data = generate_m_phate(params_per_batch, num_workers=args.workers)
    print(f"MPHATE shape: {m_phate_data.shape}")
    create_phate_visualization(
        params_per_batch,
        m_phate_data,
        filename="phate-param.png",
    )
    wandb.log({"phate-param": wandb.Image("phate-param.png")})
    print(f"Logged phate-param.png to wandb")

wandb.finish()
# END NEW ====================
