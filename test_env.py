import gym
import gym_super_mario_bros

def test_render():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")  # or 'v3'
    obs = env.reset()
    for _ in range(200):
        env.render()                # attempt to show color window
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    test_render()
