# Startrek 

## Contexte

Le but du projet Startrek est d'entrainer deux types d'algorithmes via "l'apprentissage renforcé". L'idée est de faire atterrir un module lunaire correctement sur 100 essais consécutif. Pour être précis, le but est d'obtenir une moyenne de 200 points de récompenses sur 100 essais avec les deux types d'algorithmes de notre module 

## Modèle d'entrainement

Les algorithmes que nous allons faire sont le DQN et le PPO. Les deux poursuivent le même objectif, mais c'est leur apprentissage qui sera différemment.

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