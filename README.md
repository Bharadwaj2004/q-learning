# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.


## Q LEARNING ALGORITHM
Step 1: Initialize Q-table and hyperparameters.

Step 2: Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

Step 3: After training, derive the optimal policy from the Q-table.

Step 4: Implement the Monte Carlo method to estimate state values.

Step 5: Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
```python
print('Name: B.Venkata bharadwaj')
print('Reg.No: 212222240020')
def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
![Screenshot 2024-05-09 212529](https://github.com/Bharadwaj2004/q-learning/assets/119560345/d10d8223-430a-4c50-892d-70fd27baba55)

![Screenshot 2024-05-09 212614](https://github.com/Bharadwaj2004/q-learning/assets/119560345/b2812b4e-04b0-4cfb-9ec3-7802469c704a)

![Screenshot 2024-05-09 212631](https://github.com/Bharadwaj2004/q-learning/assets/119560345/a2237feb-e3d5-4051-97fb-754d62d40195)

![Screenshot 2024-05-09 212730](https://github.com/Bharadwaj2004/q-learning/assets/119560345/9b379948-55b5-4aa2-b56f-80adba2d8d1e)


![Screenshot 2024-05-09 212748](https://github.com/Bharadwaj2004/q-learning/assets/119560345/2723f8c9-f6c9-4363-bf7b-6bb446c90ee3)

## RESULT:

Therefore a python program has been successfully developed to find the optimal policy for the given RL environment using Q-Learning and compared the state values with the Monte Carlo method.
