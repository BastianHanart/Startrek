import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

##############################################################
################## 1. ARCHITECTURE DU DQN ####################
##############################################################

class DQN(nn.Module):
    def __init__(self, state_size=8, action_size=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.network(state)

##############################################################
######################### 2. EPISODE #########################
##############################################################

def run_episode(env, model, episode_number):

    state, info = env.reset()

    total_reward = 0.0
    steps = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        state_tensor = torch.as_tensor(state)

        q_values = model(state_tensor)
        action = q_values.argmax().item()

        state, reward, terminated, truncated, info = (env.step(action))

        total_reward += reward
        steps += 1

    print(f"Episode {episode_number}/100 | Score : {total_reward:.2f} | Pas : {steps} | Termine : {terminated}| Interrompu : {truncated}")

    return total_reward

##############################################################
################### 3. PROGRAMME PRINCIPAL ###################
##############################################################

model = DQN()
model.load_state_dict(torch.load("lunarlander_dqn.pth"))

model.eval()

print("Debut de l'evaluation sur 100 episodes.\n")

rewards = []

env = gym.make("LunarLander-v3")

for episode in range(1, 101):
    rewards.append(run_episode(env, model, episode))

env.close()

##############################################################
####################### 4. BILAN FINAL #######################
##############################################################

mean_reward = np.mean(rewards)
best_reward = np.max(rewards)
worst_reward = np.min(rewards)

print("\n" + "=" * 45)
print("BILAN FINAL")
print("=" * 45)

print(f"Nombre d'episodes : {len(rewards)}")
print(f"Score moyen       : {mean_reward:.2f}")
print(f"Meilleur score    : {best_reward:.2f}")
print(f"Pire score        : {worst_reward:.2f}")

if mean_reward >= 200:
    print("\nOBJECTIF ATTEINT : score moyen >= 200 !")
else:
    print("\nOBJECTIF NON ATTEINT : score moyen < 200.")

print("=" * 45)