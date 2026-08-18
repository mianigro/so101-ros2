import os
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.optim.lr_scheduler import StepLR
from ..utils import PPOMemory
from ..utils import PPOMemoryContinuous
from ..models import TransformerModel
from ..models import MambaCVModel


class PPOAgent:
    def __init__(
        self,
        lookback_frames,
        d_model,
        num_layers,
        num_heads,
        dropout,
        input_shape,
        d_ff,
        num_actions,
        alpha,
        lr_decay,
        lr_decay_step_size,
        entropy_coef,
        min_entropy_coef,
        entropy_decay,
        value_clip_range,
        gamma,
        gae_lambda,
        policy_clip,
        observation_space,
        batch_size,
        N,
        n_epochs,
        discrete,
        architecture,
        final_layer_pooling,
        final_pool_skip,
        max_norm,
        log_clamp_lower,
        log_clamp_upper,
    ):
        # PPO hyperparams
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.N = N
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef
        self.min_entropy_coef = min_entropy_coef
        self.entropy_decay = entropy_decay
        self.max_norm = max_norm

        # Game information
        self.discrete = discrete

        # Select model type
        if architecture == "transformer":
            self.actor_critic = TransformerModel(
                lookback_frames=lookback_frames,
                input_shape=input_shape,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                dropout=dropout,
                num_actions=num_actions,
                discrete=discrete,
                final_layer_pooling=final_layer_pooling,
                final_pool_skip=final_pool_skip,
                log_clamp_lower=log_clamp_lower,
                log_clamp_upper=log_clamp_upper,
            )

        elif architecture == "mamba":
            self.actor_critic = MambaCVModel(
                lookback_frames=lookback_frames,
                input_shape=input_shape,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                num_layers=num_layers,
                num_actions=num_actions,
                discrete=discrete,
                final_layer_pooling=final_layer_pooling,
                final_pool_skip=final_pool_skip,
                log_clamp_lower=log_clamp_lower,
                log_clamp_upper=log_clamp_upper,
            )

        if discrete:
            self.memory = PPOMemory(batch_size, N, observation_space, num_actions)
        else:
            self.memory = PPOMemoryContinuous(
                batch_size, N, observation_space, num_actions
            )

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.actor_critic.parameters(),
            lr=alpha,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.scheduler = StepLR(
            self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay
        )

    def save_model(self, save_dir, step_count, group_id):
        """
        Save the model as a .pth based on the training information

        Args:
            save_dir (_type_): Location of save
            step_count (_type_): Global step count
            group_id (_type_): Timestamp of when training started
        """
        model_path = os.path.join(save_dir, f"model_step_{group_id}_{step_count}.pth")
        torch.save(self.actor_critic.state_dict(), model_path)

    def load_model(self, model_path):
        """
        Load a previously saved model from a .pth file

        Args:
            model_path (str): Path to the saved model file

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return False

            # Load state dict
            self.actor_critic.device = "cuda"
            state_dict = torch.load(model_path, map_location=self.actor_critic.device)

            # Check if the state dict has "module." prefix (from DataParallel/DistributedDataParallel)
            if all(k.startswith("module.") for k in state_dict.keys()):
                # Remove the "module." prefix
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # Apply state dict to the actor-critic model
            self.actor_critic.load_state_dict(state_dict)

            print(f"Model successfully loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def compute_gae(
        self, rewards: np.array, values: np.array, dones: np.array, last_value: float
    ) -> tuple[np.array, np.array]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_advantage = 0

        # Use the provided last_value for bootstrapping
        next_value = last_value

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value  # From last_value argument
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            running_advantage = (
                delta
                + self.gamma * self.gae_lambda * running_advantage * next_non_terminal
            )
            advantages[t] = running_advantage
            returns[t] = running_advantage + values[t]

        return advantages, returns

    def choose_action(
        self, observation: np.array
    ) -> tuple[np.array, np.array, np.array]:
        """
        Pick an action from the model

        Args:
            observation (np.array): State observation, lookback_frames amount of frames that is normalised

        Returns:
            tuple[np.array, np.array, np.array]: Action to pick, log_probs for action and value
        """
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.actor_critic.device)

        # Take action
        dist_params, value = self.actor_critic(state)

        # Handle action for each action space
        if self.discrete:
            dist = Categorical(logits=dist_params)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = np.array(action.item(), dtype=np.int32)

        else:
            mean, log_std = dist_params.chunk(2, dim=-1)
            dist = Normal(mean, torch.exp(log_std))
            action = dist.sample()
            log_prob = dist.log_prob(action)

            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1)

            action = action.cpu().numpy().flatten()
            log_prob = log_prob.item()

        return action, log_prob, value.item()

    def remember(
        self,
        state: np.array,
        action: np.array,
        probs: np.array,
        vals: np.array,
        reward: np.array,
        done: np.array,
    ) -> None:
        """
        Store a memory

        Args:
            state (np.array): State, lookback_frames amount of frames that is normalised
            action (np.array): Action picked
            probs (np.array): Probs of the action
            vals (np.array): Value from model
            reward (np.array): Reward for the action
            done (np.array): Dones
        """
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learn(self, last_value, stream_tuple=None):
        """
        Learn function

        Returns:
            Training information
        """
        # Get all data from memory
        (
            state_arr,
            action_arr,
            old_prob_arr,
            vals_arr,
            reward_arr,
            dones_arr,
            batches,
        ) = self.memory.generate_batches()

        # Compute GAE before epochs
        advantages, returns = self.compute_gae(
            reward_arr, vals_arr, dones_arr, last_value
        )

        # Convert to tensors
        advantages = torch.tensor(advantages).to(self.actor_critic.device)
        returns = torch.tensor(returns).to(self.actor_critic.device)
        vals_tensor = torch.tensor(vals_arr).to(self.actor_critic.device)

        # Training tracking
        learn_actor_loss = []
        learn_critic_loss = []
        learn_total_loss = []
        learn_grad_norms = []
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Training epochs for learn iter
        for i_batch in range(self.n_epochs):
            # Shuffle indices for each epoch
            indices = np.random.permutation(len(state_arr))
            batches = [
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]

            # Train on batches
            batch_n = 0
            for batch in batches:
                if stream_tuple:
                    frame1, frame2, stream_client = stream_tuple

                    stream_client.send_frame_training(
                        frame1,
                        frame2,
                        i_batch,
                        self.n_epochs,
                        batch_n,
                        self.N // self.batch_size,
                    )

                # Get batch data
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(
                    self.actor_critic.device
                )
                old_probs = torch.tensor(old_prob_arr[batch]).to(
                    self.actor_critic.device
                )
                actions = torch.tensor(action_arr[batch]).to(self.actor_critic.device)
                batch_returns = returns[batch]
                batch_advantages = advantages[batch]
                batch_values = vals_tensor[batch]

                # Normalize advantages per batch
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )

                # Forward pass
                dist_params, state_values = self.actor_critic(states)

                # Selection action based on action space
                if self.discrete:
                    dist = Categorical(logits=dist_params)
                    entropy = dist.entropy().mean()
                    new_probs = dist.log_prob(actions)

                else:
                    mean, log_std = dist_params.chunk(2, dim=-1)
                    dist = Normal(mean, torch.exp(log_std))

                    # For each action dimension, get the log prob
                    new_probs = dist.log_prob(actions)

                    # Must sum across action dimensions before comparing with old_probs
                    new_probs = new_probs.sum(-1)

                    # Entropy calculation
                    entropy_per_dim = dist.entropy()
                    entropy = entropy_per_dim.mean()

                # Policy loss
                #   Calculates the difference between the log probability of the action under the new policy
                #   (new_probs) and the log probability of the action under the old policy (old_probs), and
                #   then exponentiates the result to obtain the probability ratio.
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * batch_advantages
                )

                # Actor loss is probability ratio * advantages
                #   When A > 0 (good action), we want r(θ) > 1 to increase the probability of this action.
                #   When A < 0 (bad action), we want r(θ) < 1 to decrease the probability of this action.
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Value loss - determines value of the state in terms of future expected reward
                #   This loss function guides the training of the critic (value function) network using squared error.
                #   By minimizing this loss, we're teaching the critic to accurately predict the expected returns for
                #   given states.
                value_pred = state_values.squeeze()
                value_target = batch_returns
                value_loss_unclipped = (value_pred - value_target).pow(2)
                value_clipped = batch_values + torch.clamp(
                    value_pred - batch_values,
                    -self.value_clip_range,
                    self.value_clip_range,
                )
                value_loss_clipped = (value_clipped - value_target).pow(2)

                # Critic loss is MSE of return - critic value
                #   Error square is above, mean is below here
                critic_loss = (
                    0.5 * torch.min(value_loss_unclipped, value_loss_clipped).mean()
                )

                # Total loss with entropy regularisation
                self.entropy_coef = max(self.entropy_coef, self.min_entropy_coef)
                total_loss = (
                    actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), max_norm=self.max_norm
                )

                # Step optimiser
                self.optimizer.step()

                # Tracking loss
                learn_actor_loss.append(actor_loss.detach().cpu().item())
                learn_critic_loss.append(critic_loss.detach().cpu().item())
                learn_total_loss.append(total_loss.detach().cpu().item())
                learn_grad_norms.append(grad_norm.item())

                batch_n += 1

        # Step scheduler
        self.scheduler.step()

        # Step entropy
        self.entropy_coef_report = self.entropy_coef
        self.entropy_coef = self.entropy_coef * self.entropy_decay

        # Clear memory after all epochs
        self.memory.clear_memory()

        return (
            np.mean(learn_actor_loss),
            np.mean(learn_critic_loss),
            np.mean(learn_total_loss),
            entropy.item(),
            self.entropy_coef_report * entropy,
            current_lr,
            np.mean(learn_grad_norms),
        )
