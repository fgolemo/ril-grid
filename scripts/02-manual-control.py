import gym
from gym_minigrid.window import Window


def redraw(img):
    img = env.render("rgb_array", tile_size=32)
    window.show_img(img)


def reset():
    # env.seed(123)

    obs = env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    obs, reward, done, info = env.step(action)
    print("step=%s, reward=%.2f" % (env.step_count, reward))

    if done:
        print("done!")
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset()
        return

    if event.key == "left":
        step(env.actions.left)
        return
    if event.key == "right":
        step(env.actions.right)
        return
    if event.key == "up":
        step(env.actions.forward)
        return

    # Spacebar
    # if event.key == " ":
    #     step(env.actions.toggle)
    #     return
    # if event.key == "pageup":
    #     step(env.actions.pickup)
    #     return
    if event.key == " ":  # press space to pick up item
        step(env.actions.pickup)
        return
    # if event.key == "pagedown":
    #     step(env.actions.drop)
    #     return
    #
    # if event.key == "enter":
    #     step(env.actions.done)
    #     return


import ril_grid  # for registering the env

env = gym.make("Ril-Test1-v1")


window = Window("gym_minigrid - Ril-Test1-v0")
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
