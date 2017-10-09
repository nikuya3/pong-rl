from pong import get_env, get_net, process_observation

env = get_env()
net = get_net()
env.reset()
eta = 1e-5
gamma = 0.99
render = False
rollouts = 0
actions = [0]
observations = []
current_gradients = []
while True:
    action = actions[-1] + 1  # Actions: 1 -> Nothing, 2 -> Up, 3 -> Down
    observation, reward, done, info = env.step(action)
    if render:
        env.render()
    observation = process_observation(observation)
    observations.append(observation)
    scores = net.forward_pass(observation)
    # Bias the policy to take the sampled action -> to take clear actions in general
    y = np.argmax(scores)
    dscores = np.full(len(scores), -1)
    dscores[y] = 1
    current_gradients.append(dscores)
    actions.append(y)
    if len(actions) == 10:
        from PIL import Image
        Image.fromarray(observation).show()
        Image.fromarray(observation - observations[-1]).show()
    if done:
        grads = np.array(current_gradients) * reward
        xs = np.array(observations)
        net.backward_pass(xs, grads)
        net.update_parameters(eta)
        current_gradients = []
        observations = []
        rollouts += 1
        if rollouts % 50 == 0:
            net.save()
        env.reset()
        print("Episode", rollouts, "Reward:", reward)