class Storage():
    def __init__(self):
        self.reset()

    def add(self, state, action, reward, policy_prob, values, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policy_probs.append(policy_prob)
        self.values.append(values)
        self.dones.append(done)

    def get_all(self):
        return (self.states,
                self.actions,
                self.rewards,
                self.policy_probs,
                self.dones)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.policy_probs = []
        self.values = []
        self.dones = []
