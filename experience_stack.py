class Memory:
    """
    Memory queue that is used to store transitions in form (s0, a, r, s1).
    """
    def __init__(self, max_length=100000):
        self.queue = []
        self.max_length = max_length
        self.number_of_items = 0

    def push(self, transition):
        if len(self.queue) > self.max_length:
            self.pop()
        self.queue.append(transition)
        self.number_of_items = len(self.queue)

    def pop(self):
        if len(self.queue) > 0:
            self.queue.pop()
            self.number_of_items = len(self.queue)

    def clear(self):
        self.queue = []
        self.number_of_items = 0


