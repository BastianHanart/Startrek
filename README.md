<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/JupyterLab-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="JupyterLab" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Matplotlib-37BF2E?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
</div>

# Startrek 

## Utilisation du projet

Pour utiliser le projet, il faut lancer un terminal à la racine du projet et lancer cette commande :

```bash
python model.py ALGO MODEL [EPISODES]
```
avec : 

- ALGO = l'algorithme à utiliser (PPO ou DQN)
- MODEL = Modèle sauvegardé dans le fichier PTH (ex : PTH/lunarlander_dqn_seed_42.pth)
- [EPISODES] = Nombre d'épisodes à tester (Paramètre non obligatoire, si non renseigné, 1 épisode joué)

Un exemple de commande serai : 

```bash
python model.py DQN PTH/lunarlander_dqn_seed_42.pth 10
```

Pour plus d'informations : 

```bash
python model.py -h
```

## Contexte

Le but du projet Startrek est d'entraîner deux types d'algorithmes via "l'apprentissage renforcé". L'idée est de faire atterrir un module lunaire correctement sur 100 essais consécutif. Pour être précis, le but est d'obtenir une moyenne de 200 points de récompense sur 100 essais avec les deux types d'algorithmes de notre module

## Modèle d'entrainement

Les algorithmes que nous allons faire sont le DQN et le PPO. Les deux poursuivent le même objectif, mais c'est leur apprentissage qui sera différent.

### DQN - Deep Q-Network
Le DQN apprend à estimer la valeur des différentes actions possibles dans un état donné.

Il utilise :
- Un réseau neuronal.
- Une mémoire d'expériences appelée replay buffer.
- Une stratégie d'exploration appelée epsilon-greedy.
- Un second réseau appelé target network.
- Une fonction de perte basée sur l'erreur quadratique moyenne.

Il permet de répondre à la problématique suivante : 
" Dans cette situation, quelle action semble la plus intéressante pour obtenir le meilleur résultat à long terme ? "

### PPO - Proximal Policy Optimization

Le PPO apprend directement une politique de décision, c'est-à-dire une manière de choisir les actions en fonction de l'état actuel.

Il utilise notamment :
- Un acteur, appelé actor, qui choisit les actions.
- Un critique, appelé critic, qui estime la valeur des états.
- Un stockage temporaire des expériences appelé rollout storage.
- Un calcul des avantages, ou advantages.
- Une fonction de perte combinant plusieurs termes.
- Une méthode de mise à jour qui limite les changements trop importants de politique.

Celui-ci permet de répondre à la problématique :
" Dans cette situation, quelles actions dois-je privilégier, et avec quelle probabilité ? "

## Lancer les modèles

Pour lancer soit le DQN, soit le PPO, pour voir leur fonctionnement et tester d'autres seeds, paramètres, etc. c'est la même méthode. Il suffit de se rendre dans le fichier [dqn](dqn.ipynb) ou dans le fichier [ppo](ppo.ipynb).
Une fois à l'intérieur, faites ces deux commandes dans le terminal :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Ensuite, il vous suffit d'appuyer sur le bouton "Run All" en haut de la fenêtre.
Il vous faudra attendre que toutes les cellules soient exécutées, ce qui peut prendre plus ou moins de temps selon le nombre de seed que vous testez.
Lorsque vous voyez dans chaque cellule, en bas à gauche, une petite validation, cela veut dire que tout est fini et vous pouvez donc regarder les résultats obtenus.  