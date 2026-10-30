import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

##############################################################
################## 1. ARCHITECTURE DU MODELE #################
##############################################################

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

# on recrée la structure de l'agent pour que PyTorch puisse charger l'objet
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)

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
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            # _ car on garde juste juste les actions
            action_logits, _ = model(state_tensor)
            
        # action avec la plus grande proba
        action = action_logits.argmax().item()

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    print(f"Episode {episode_number}/100 | Score : {total_reward:.2f} | Pas : {steps} | Termine : {terminated} | Interrompu : {truncated}")

    return total_reward

##############################################################
################### 3. PROGRAMME PRINCIPAL ###################
##############################################################

state_dim = 8
action_dim = 4

# 1. On charge l'objet PPOAgent complet (weights_only=False est obligatoire ici)
loaded_agent = torch.load("lunarlander_ppo.pth", map_location=torch.device('cpu'), weights_only=False)

# extraire le réseau de neurone
model = loaded_agent.policy

# début évalutation
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