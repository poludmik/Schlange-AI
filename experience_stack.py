class Transition:
    def __init__(self, state0, action, reward, state1, Q_value):
        self.state0 = state0
        self.action = action
        self.reward = reward
        self.state1 = state1
        self.Q_value = Q_value

class Memory:
    def __init__(self, max_length=100000):
        self.stack = []
        self.max_length = max_length
        self.number_of_items = 0

    def push(self, transition):
        if len(self.stack) > self.max_length:
            self.pop()
        self.stack.append(transition)
        self.number_of_items = len(self.stack)

    def pop(self):
        if len(self.stack) > 0:
            self.stack.pop()
            self.number_of_items = len(self.stack)

    def clear(self):
        self.stack = []
        self.number_of_items = 0


