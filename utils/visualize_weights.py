import numpy as np
import m_phate
import scprep
import argparse

# BEGIN NEW ====================
def generate_m_phate(data, num_workers=20):
    # embedding
    m_phate_op = m_phate.M_PHATE(n_jobs=num_workers)
    m_phate_data = m_phate_op.fit_transform(data)
    print(f"m-PHATE done")
    return m_phate_data


def create_phate_visualization(data, m_phate_data, filename="phate-param.png"):
    shape_data = data.shape
    time = np.repeat(np.arange(shape_data[0]), shape_data[1])
    print(f"Plotting ...")
    plot = scprep.plot.scatter2d(
        m_phate_data,
        c=time,
        ticks=False,
        label_prefix="M-PHATE",
        filename=filename,
        dpi=600,
        title="Learnable Weigths per Time Step",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="yelp_clean", help="choose the dataset"
    )

    parser.add_argument(
        "--model_type", type=str, default="T-DiffRec", help="type DRS Model"
    )

    parser.add_argument(
        "--run_name", type=str, default="", help="run name extension for wandb"
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--workers", type=int, default=10, help="number of workers for m-phate"
    )
    args = parser.parse_args()

    data = np.load(
        f"mPHATE/{args.model_type}_{args.dataset}_{args.seed}_{args.run_name}_params_per_batch.npy"
    )
    print(f"Data shape: {data.shape}")  # n_time_steps, n_points, n_dim
    m_phate_data = generate_m_phate(data, num_workers=args.workers)
    create_phate_visualization(
        data,
        m_phate_data,
        filename=f"mPHATE/{args.model_type}_{args.dataset}_{args.seed}_{args.run_name}_params_per_batch.png",
    )


if __name__ == "__main__":
    main()
# END NEW ====================
