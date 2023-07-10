import enum
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    LEARNABLE_PARAM = enum.auto()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        mean_type,
        noise_schedule,
        noise_scale,
        noise_min,
        noise_max,
        steps,
        device,
        history_num_per_term=10,
        beta_fixed=True,
        attention_weighting=False,
        item_embeddings=None,
    ):
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(
            steps, history_num_per_term, dtype=torch.float64
        ).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        self.attention_weighting = attention_weighting
        try:
            self.item_embeddings = item_embeddings.to(self.device)
        except AttributeError:
            self.item_embeddings = None


        if noise_scale != 0.0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(
                self.device
            )
            if beta_fixed:
                self.betas[
                    0
                ] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert (
                len(self.betas) == self.steps
            ), "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (
                self.betas <= 1
            ).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(
                    self.steps, np.linspace(start, end, self.steps, dtype=np.float64)
                )
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        elif (
            self.noise_schedule == "binomial"
        ):  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        ).to(
            self.device
        )  # alpha_{t-1}
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]
        ).to(
            self.device
        )  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                [self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]
            )
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.0:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = (
                    out["mean"]
                    + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                )
            else:
                x_t = out["mean"]
        return x_t

    def calculate_user_interaction_embedding(self, x_t):
        # create mask to only consider the interactions of a user
        interacted_indices = x_t > 0

        # create mask to only consider the interactions of a user
        interacted_indices = interacted_indices.nonzero()

        user_interactions = torch.bincount(
            interacted_indices[:, 0]
        )  # nr of interactions each user has
        # prepare storage
        len_interactions = x_t.shape[1]
        max_user_interactions = torch.max(user_interactions)
        print(f"max nr of interactions per user: {max_user_interactions}")
        # to store item embedding instead of the item a user has interacted with
        all_user_interaction_embeddings = torch.empty(
            x_t.shape[0],
            len_interactions,
            self.item_embeddings.shape[1],
            device=self.device,
        )
        # to store the last valid interaction embedding of a user
        last_user_interaction_embedding = torch.empty(
            x_t.shape[0],
            self.item_embeddings.shape[1],
            device=self.device,
        )

        for user in range(x_t.shape[0]):
            # get the interactions with items for a user
            user_interacted_indices = interacted_indices[
                interacted_indices[:, 0] == user
            ]
            # get the item embeddings corresponding to the interactions
            user_interaction_embeddings = self.item_embeddings[
                user_interacted_indices[:, 1], :
            ]

            # keep only interaction indices as list
            user_interacted_indices = user_interacted_indices[:, 1].tolist()

            if len(user_interaction_embeddings) > 0:  # if user has interactions
                interaction_count = 0

                # all possible interactions
                for interaction in range(len_interactions):
                    if interaction in user_interacted_indices:
                        # for each interaction of a user, add the interaction embeddings, and store last valid interaction embedding
                        all_user_interaction_embeddings[
                            user, interaction
                        ] = user_interaction_embeddings[
                            interaction_count
                        ]  # m_t

                        interaction_count += 1

                # store last valid interaction embedding
                last_user_interaction_embedding[user] = user_interaction_embeddings[-1]

        # all_user_interaction_embeddings = torch.nan_to_num(
        #     all_user_interaction_embeddings
        # )

        # calculate mean of the embeddings but do not include nan values
        mean_user_interacted_embeddings = torch.nanmean(
            all_user_interaction_embeddings, dim=1
        )

        # ensure same dimensionalities
        last_user_interaction_embedding = last_user_interaction_embedding.unsqueeze(
            dim=1
        )
        mean_user_interacted_embeddings = mean_user_interacted_embeddings.unsqueeze(
            dim=1
        )

        return (
            all_user_interaction_embeddings,
            last_user_interaction_embedding,
            mean_user_interacted_embeddings,
        )

    def training_losses(
        self,
        model,
        x_start,
        reweight=False,
    ):
        """
        # NEW ================================
        self: Diffusion Model
        Model: DNN
        x_start:  batch from data loader, bs x n_items ([400, 2810])
        ts: [400], int
        pt: [400, float
        x_t: 400, 2810: float, incl. negative values
        pt: [400], float
        model_output: [400, 2810], float, incl. negative values
        weights: [400], float
        # END NEW ================================
        """

        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, "importance", model)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0.0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        # NEW ================================
        if self.attention_weighting:
            """attention weighting"""
            if self.mean_type == ModelMeanType.START_X:
                # get the embeddings of all items interacted with (x), and calculate the mean embedding per user (batch element)

                (
                    all_user_interaction_embeddings,
                    last_user_interaction_embedding,
                    mean_user_interacted_embeddings,
                ) = self.calculate_user_interaction_embedding(x_t)

                # forward pass with attention weighting
                model_output = model(
                    x_t,
                    ts,
                    all_user_interaction_embeddings=all_user_interaction_embeddings,
                    last_user_interaction_embedding=last_user_interaction_embedding,
                    mean_user_interacted_embeddings=mean_user_interacted_embeddings,
                )
            # END NEW ================================
            else:
                raise NotImplementedError

        else:
            #  OLD: forward pass without attention weighting
            model_output = model(x_t, ts)  # DNN

        terms = {}
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.LEARNABLE_PARAM: x_start,
        }[self.mean_type]

        # print(f"Model output shape: {model_output.shape}")
        # print(f"model output: {model_output}")
        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / (
                    (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, model_output))
                    ** 2
                    / 2.0
                )
                loss = torch.where((ts == 0), likelihood, mse)

            # NEW ================================
            elif self.mean_type == ModelMeanType.LEARNABLE_PARAM:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            # END NEW ================================

        else:
            weight = torch.tensor([1.0] * len(target)).to(device)
            loss = mse

        terms["loss"] = weight * loss
        # print(f"Shape of the weights: {weight.shape}")

        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    def sample_timesteps(
        self, batch_size, device, method="uniform", model=None, uniform_prob=0.001
    ):
        if method == "importance":  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method="uniform")

            # NEW ================================
            if self.mean_type == ModelMeanType.LEARNABLE_PARAM:
                new_hist = model.param * self.Lt_history.clone()
                Lt_sqrt = torch.sqrt(torch.mean(new_hist**2, axis=-1))
            # END NEW ================================

            Lt_sqrt = torch.sqrt(
                torch.mean(self.Lt_history**2, axis=-1)
            )  # original code
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1.0 < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            # NEW ================================
            # store the param values per batch
            model.param_storage.append(model.param.detach().clone())
            # END NEW ============================
            return t, pt

        elif method == "uniform":  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + self._extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(
            x,
            t,
        )

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        # BEGIN NEW ====================
        elif self.mean_type == ModelMeanType.LEARNABLE_PARAM:
            pred_xstart = model_output
        # END NEW ======================
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * eps
        )

    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
