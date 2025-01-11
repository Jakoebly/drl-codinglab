from collections import deque

# Mocking the MultiStepExtension and Agent for testing
class MockAgent:
    def __init__(self, gamma, n_steps):
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)

class MockMultiStepExtension:
    def __init__(self, agent):
        self.agent = agent

    def _get_n_step_transition(self):
        multi_step_reward = 0
        for idx, (s, a, r, ns, d) in enumerate(self.agent.n_step_buffer):
            multi_step_reward += (self.agent.gamma ** idx) * r
            if d:
                return multi_step_reward, (s, a, r, ns, d), True
        return multi_step_reward, self.agent.n_step_buffer[-1], False

# Parameters for testing
gamma = 0.93  # Discount factor
n_steps = 3  # Multi-step size
agent = MockAgent(gamma, n_steps)
multi_step_extension = MockMultiStepExtension(agent)

# Test cases
test_cases = [
    # Format: (input_transitions, expected_cumulative_reward, expected_done_flag)
    (
        [(0, 0, 1.0, 1, False), (1, 1, 1.0, 2, False), (2, 2, 1.0, 3, False)],
        1.0 + gamma * 1.0 + gamma ** 2 * 1.0,
        False,
    ),
    (
        [(0, 0, 1.0, 1, False), (1, 1, 1.0, 2, True), (2, 2, 1.0, 3, False)],
        1.0 + gamma * 1.0,
        True,
    ),
    (
        [(0, 0, 1.0, 1, True), (1, 1, 1.0, 2, False), (2, 2, 1.0, 3, False)],
        1.0,
        True,
    ),
]

# Run tests
for idx, (transitions, expected_reward, expected_done) in enumerate(test_cases):
    agent.n_step_buffer.extend(transitions)
    reward, transition, done = multi_step_extension._get_n_step_transition()
    assert abs(reward - expected_reward) < 1e-5, f"Test {idx} failed: expected reward {expected_reward}, got {reward}"
    assert done == expected_done, f"Test {idx} failed: expected done {expected_done}, got {done}"
    print(f"Test {idx} passed!")
