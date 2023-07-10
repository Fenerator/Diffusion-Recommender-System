import torch
import torch as th

embeddings = torch.tensor([[1, 2, 3], [3, 4, 7], [5, 6, 7], [7, 8, 7], [9, 10, 7]])
print(f"embeddings {embeddings}")

x_start = torch.tensor(
    [[166, 155, 167, 167, 167], [11, 12, 13, 14, 15], [166, 2, 3, 4, 15]]
)
print(f"x_start {x_start}")


interacted_indices = x_start > 153  # TODO define threshold

interacted_indices = interacted_indices.nonzero()
nr_user_interactions = torch.bincount(
    interacted_indices[:, 0]
)  # nr of interactions each user has


# prepare storage
max_len = torch.max(nr_user_interactions)
print(f"max nr of interactions per user: {max_len}")
all_user_interaction_embeddings = torch.empty(
    x_start.shape[0], max_len, embeddings.shape[1]
)  # store each interaction embeding on its own

# store last no-zero interaction embedding for each user
last_user_interaction_embedding = torch.empty(x_start.shape[0], embeddings.shape[1])

for user in range(x_start.shape[0]):
    user_interacted_indices = interacted_indices[interacted_indices[:, 0] == user]
    interaction_embeddings = embeddings[user_interacted_indices[:, 1], :]

    print(f"User: {user}: nr of interactions: {len(interaction_embeddings)}")

    if len(interaction_embeddings) > 0:
        for interaction in range(len(interaction_embeddings)):
            all_user_interaction_embeddings[user, interaction] = interaction_embeddings[
                interaction
            ]  # m_t

        # add last interaction embedding to
        print(f"Last valid interaction embedding: {interaction_embeddings[-1]}")
        last_user_interaction_embedding[user] = interaction_embeddings[-1]


all_user_interaction_embeddings = torch.nan_to_num(all_user_interaction_embeddings)
print(
    f"all_user_embeddings: {all_user_interaction_embeddings}\n shape: {all_user_interaction_embeddings.shape}"
)

# calculate mean of the embeddings but do not inclued nan values
mean_user_interacted_embeddings = torch.nanmean(all_user_interaction_embeddings, dim=1)
print(f"mean_user_interacted_embeddings shape: {mean_user_interacted_embeddings.shape}")
print(f"last_user_interaction_embedding: {last_user_interaction_embedding}")
