import sys
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
################### 2. ARCHITECTURE DU PPO ###################
##############################################################

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()


        # ce réseau va permettre de donner une probabiltié de la prochaine action a réaliser selon l'env du vaisseau
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), # prend nos 8 valeurs d'entrées et les envoie dans une couche de 64 neuronnes
            nn.Tanh(), # convertit nos donneés entre -1 et 1 pour une meilleur exploitation 
            nn.Linear(64, 64), # on refournit nos données déjà entrainé sur nos 64 neuronnes dnas 64 nouveaux neurones une couche de plus
            nn.Tanh(),
            nn.Linear(64, action_dim) # output des 4 actions
        )
        
        # sort une seule valeur estimation de l'action
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) 
        )
        

    #  cet fonction permet de choisir le chemin que la donnée va suivre dans notre réseau de neuronne
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

##############################################################
################### 3. CHARGEMENT DU MODEL ###################
##############################################################

def load_model(ALGO, MODEL):

    env = gym.make("LunarLander-v3")

    if ALGO.lower() == "dqn":
        model = DQN()

    elif ALGO.lower() == "ppo":
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n)

    else:
        raise ValueError(f"Algorithme {ALGO} inconnu, essayé 'DQN' ou 'PPO'")

    state_dict = torch.load(MODEL, weights_only=False)

    model.load_state_dict(state_dict)
    model.eval()

    return model


##############################################################
######################### 4. EPISODE #########################
##############################################################

def run_episode(env, model, episode_number, total_episodes):

    state, info = env.reset()

    total_reward = 0.0
    steps = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            if algo.lower() == "dqn":

                q_values = model(state_tensor)
                action = q_values.argmax().item()

            elif algo.lower() == "ppo":

                action_probs, state_value = model(state_tensor)

                # On choisit l'action la plus probable pour une évaluation déterministe
                action = action_probs.argmax().item()

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    print(f"Episode {episode_number}/{total_episodes} "f"| Score : {total_reward:.2f} " f"| Pas : {steps} " f"| Termine : {terminated} " f"| Interrompu : {truncated}")

    return total_reward


##############################################################
################### 5. PROGRAMME PRINCIPAL ###################
##############################################################

def main(algo, model_path, episodes):

    model = load_model(algo, model_path)

    env = gym.make("LunarLander-v3", render_mode = "human")
    rewards = []

    try:
        for episode in range(1, episodes + 1):
            reward = run_episode(env, model, episode, episodes)
            rewards.append(reward)

    finally:
        env.close()

    return rewards


##############################################################
####################### 6. BILAN FINAL #######################
##############################################################

def display_results(rewards):

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


##############################################################
######################## 7. EXECUTION ########################
##############################################################

if __name__ == "__main__":

    try:

        if len(sys.argv) == 2 and sys.argv[1] == "-h":
            print("Usage \n     python model.py ALGO MODEL [EPISODES] \n\nDESCRIPTION\n    ALGO       algorithm of the saved model: dqn or ppo \n    MODEL       path to the saved model file \n    EPISODES    number of episodes to run (default: 1)")
        
        else :
            if len(sys.argv) < 3 or len(sys.argv) > 4:
                raise ValueError("\nUsage \n     python model.py ALGO MODEL [EPISODES] \n\nDESCRIPTION\n    ALGO       algorithm of the saved model: dqn or ppo \n    MODEL       path to the saved model file \n    EPISODES    number of episodes to run (default: 1)")

            algo = sys.argv[1].lower()
            model_path = sys.argv[2]

            if len(sys.argv) == 4:
                episodes = int(sys.argv[3])  
            else :
                episodes = 1

            if episodes <= 0:
                raise ValueError("Le nombre d'episodes doit etre strictement positif et supérieur à 0.")

            rewards = main(algo, model_path, episodes)

            display_results(rewards)

    except Exception as error:
        print(f"Erreur : {error}")