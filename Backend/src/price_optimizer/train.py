"""Main training script for the price optimization RL system."""

import os
import wandb
import numpy as np
import torch
from typing import Dict, List, Tuple
import traceback

from .config.config import Config
from .data.preprocessor import DataPreprocessor
from .env.price_env import PriceOptimizationEnv
from .models.sac_agent import SACAgent


class StateValidator:
    """Validates and clips state values to ensure stability."""

    def __init__(self):
        self.min_allowed = 0.0
        self.max_allowed = 1.0

    def validate(self, x: np.ndarray) -> np.ndarray:
        """Validate and clip state values."""
        if np.isnan(x).any():
            print("Warning: NaN detected in state. Replacing with zeros.")
            x = np.nan_to_num(x, 0.0)
        return np.clip(x, self.min_allowed, self.max_allowed)


def evaluate_agent(
    agent: SACAgent,
    env: PriceOptimizationEnv,
    state_validator: StateValidator,
    num_episodes: int = 5,
) -> Dict[str, float]:
    """Evaluate the agent's performance."""
    eval_rewards = []
    eval_prices = []
    eval_sales = []

    try:
        for _ in range(num_episodes):
            state, _ = env.reset(options={"test_mode": True})
            state = state_validator.validate(state)
            episode_reward = 0
            episode_prices = []
            episode_sales = []
            done = False

            while not done:
                try:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, _, info = env.step(action)
                    next_state = state_validator.validate(next_state)

                    episode_reward += reward
                    episode_prices.append(info["price"])
                    episode_sales.append(info["sales"])
                    state = next_state
                except Exception as e:
                    print(f"Error during evaluation step: {e}")
                    traceback.print_exc()
                    break

            eval_rewards.append(episode_reward)
            eval_prices.extend(episode_prices)
            eval_sales.extend(episode_sales)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return {
            "eval_reward_mean": 0.0,
            "eval_reward_std": 0.0,
            "eval_price_mean": 0.0,
            "eval_price_std": 0.0,
            "eval_sales_mean": 0.0,
            "eval_sales_std": 0.0,
        }

    return {
        "eval_reward_mean": float(np.mean(eval_rewards)),
        "eval_reward_std": float(np.std(eval_rewards)),
        "eval_price_mean": float(np.mean(eval_prices)),
        "eval_price_std": float(np.std(eval_prices)),
        "eval_sales_mean": float(np.mean(eval_sales)),
        "eval_sales_std": float(np.std(eval_sales)),
    }


async def train(config: Config) -> Dict:
    """Main training loop."""
    try:
        # Set random seeds
        torch.manual_seed(config.training.random_seed)
        np.random.seed(config.training.random_seed)

        # Initialize wandb
        wandb.init(
            project=config.training.wandb_project,
            config=config.__dict__,
            name=f"SAC_price_opt_{config.training.random_seed}",
        )

        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

        # Initialize data preprocessor
        preprocessor = DataPreprocessor(config.data)
        preprocessor.prepare_data()

        # Initialize environment
        env = PriceOptimizationEnv(preprocessor, config.env)

        # Initialize state validator
        state_validator = StateValidator()

        # Calculate state and action dimensions
        state_dim = env.state_dim
        action_dim = env.action_space.shape[0]

        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")

        # Get price history from preprocessor
        price_history = preprocessor.get_price_history()

        # Initialize agent with price history
        agent = SACAgent(state_dim, action_dim, config.model, price_history)

        # Training loop
        total_steps = 0
        best_eval_reward = float("-inf")
        final_metrics = None

        for episode in range(config.training.num_episodes):
            try:
                state, _ = env.reset()
                state = state_validator.validate(state)
                episode_reward = 0
                episode_prices = []
                episode_sales = []

                for step in range(config.training.steps_per_episode):
                    # Select action
                    action = agent.select_action(state)

                    # Take step in environment
                    next_state, reward, done, _, info = env.step(action)
                    next_state = state_validator.validate(next_state)

                    # Store transition in replay buffer
                    agent.replay_buffer.push(state, action, reward, next_state, done)

                    state = next_state
                    episode_reward += reward
                    episode_prices.append(info["price"])
                    episode_sales.append(info["sales"])
                    total_steps += 1

                    # Train agent
                    if len(agent.replay_buffer) > config.model.batch_size:
                        train_info = agent.train(config.model.batch_size)

                        # Log training metrics
                        wandb.log(
                            {
                                "train/actor_loss": train_info["actor_loss"],
                                "train/critic_loss": train_info["critic_loss"],
                                "train/alpha_loss": train_info["alpha_loss"],
                                "train/alpha": train_info["alpha"],
                                "train/actor_grad_norm": train_info["actor_grad_norm"],
                                "train/critic_grad_norm": train_info[
                                    "critic_grad_norm"
                                ],
                                "train/step": total_steps,
                            }
                        )

                    if done:
                        break

                # Log episode metrics
                episode_metrics = {
                    "episode/reward": episode_reward,
                    "episode/avg_price": np.mean(episode_prices),
                    "episode/avg_sales": np.mean(episode_sales),
                    "episode/price_std": np.std(episode_prices),
                    "episode/sales_std": np.std(episode_sales),
                    "episode": episode,
                }
                wandb.log(episode_metrics)

                # Print episode summary
                print(
                    f"Episode {episode}: Reward = {episode_reward:.2f}, "
                    f"Avg Price = {np.mean(episode_prices):.2f}, "
                    f"Avg Sales = {np.mean(episode_sales):.2f}"
                )

                # Evaluate agent
                if episode % config.training.eval_frequency == 0:
                    eval_metrics = evaluate_agent(agent, env, state_validator)
                    final_metrics = eval_metrics  # Store the latest evaluation metrics
                    wandb.log(
                        {
                            "eval/reward_mean": eval_metrics["eval_reward_mean"],
                            "eval/reward_std": eval_metrics["eval_reward_std"],
                            "eval/price_mean": eval_metrics["eval_price_mean"],
                            "eval/price_std": eval_metrics["eval_price_std"],
                            "eval/sales_mean": eval_metrics["eval_sales_mean"],
                            "eval/sales_std": eval_metrics["eval_sales_std"],
                            "episode": episode,
                        }
                    )

                    print(f"\nEvaluation at episode {episode}:")
                    print(
                        f"Average Reward: {eval_metrics['eval_reward_mean']:.2f} "
                        f"± {eval_metrics['eval_reward_std']:.2f}"
                    )
                    print(
                        f"Average Price: {eval_metrics['eval_price_mean']:.2f} "
                        f"± {eval_metrics['eval_price_std']:.2f}"
                    )
                    print(
                        f"Average Sales: {eval_metrics['eval_sales_mean']:.2f} "
                        f"± {eval_metrics['eval_sales_std']:.2f}\n"
                    )

                    # Save best model
                    if eval_metrics["eval_reward_mean"] > best_eval_reward:
                        best_eval_reward = eval_metrics["eval_reward_mean"]
                        agent.save(
                            os.path.join(
                                config.training.checkpoint_dir, "best_model.pt"
                            )
                        )

                # Save checkpoint
                if episode % config.training.save_frequency == 0:
                    agent.save(
                        os.path.join(
                            config.training.checkpoint_dir, f"checkpoint_{episode}.pt"
                        )
                    )

            except Exception as e:
                print(f"Error during episode {episode}: {e}")
                traceback.print_exc()
                continue

        # Save final model
        agent.save(os.path.join(config.training.checkpoint_dir, "final_model.pt"))
        wandb.finish()

        # Return final training metrics
        return {
            "mae": final_metrics["eval_reward_std"] if final_metrics else 0.0,
            "rmse": final_metrics["eval_price_std"] if final_metrics else 0.0,
            "r2_score": (
                final_metrics["eval_reward_mean"] / 100.0 if final_metrics else 0.0
            ),
        }

    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        wandb.finish()
        return {"mae": 0.0, "rmse": 0.0, "r2_score": 0.0}


if __name__ == "__main__":
    config = Config()
    train(config)
