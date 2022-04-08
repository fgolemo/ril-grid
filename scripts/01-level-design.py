import time

from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, Box, COLORS, Door
from gym_minigrid.rendering import fill_coords, point_in_rect


class Shelf(Box):
    # def __init__(self, color, contains=None):
    #     super().__init__('box', color)
    #     self.contains = contains

    def can_pickup(self):
        return False

    def render(self, img):
        # c = COLORS[self.color]
        #
        # # Outline
        # fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        # fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))
        #
        # # Horizontal slit
        # fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

    def toggle(self, env, pos):
        # # Replace the box by its contents
        # env.grid.set(*pos, self.contains)
        # return True
        return False


class RilEnv(MiniGridEnv):
    def __init__(self, goal_pos=None):
        self._agent_default_pos = (5, 1)
        self._goal_default_pos = goal_pos
        self.shelves = [
            (2, 0),
            (3, 0),
            (5, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (1, 6),
            (2, 6),
            (3, 6),
            (4, 6),
        ]
        super().__init__(width=6 + 2, height=7 + 2, max_steps=40)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        for coords in self.shelves:
            coords = [x + 1 for x in coords]  # to account for walls
            self.grid.set(*coords, Shelf("purple"))

        self.grid.set(2, 1, Door("yellow", is_open=True))

        self.agent_pos = self._agent_default_pos
        # self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = self._rand_int(0, 4)  # assuming random start direction

        pos = self._rand_elem(self.shelves)
        self.goal_pos = (pos[0] + 1, pos[1] + 1)
        self.grid.set(*self.goal_pos, Goal())

        self.mission = "Pick up the target item and leave"

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


env = RilEnv()


def print_obs(obs):
    img = obs["image"]
    for y in range(img.shape[0]):
        print("".join([str(x) for x in img[img.shape[0] - 1 - y, :, 0]]))
    print()


while True:
    obs = env.reset()
    img = obs["image"]
    print("=== reset,", img.shape, img.dtype, img.max(), img.max())
    print_obs(obs)
    env.render("human")
    done = False
    while not done:
        act = env.action_space.sample()
        obs, rew, done, misc = env.step(act)
        print_obs(obs)
        env.render("human")
        print(f"action: {act}, rew {rew}, done {done}, misc {misc}")
        time.sleep(10)
