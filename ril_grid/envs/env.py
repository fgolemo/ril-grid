from enum import IntEnum

import cv2
import numpy as np

from gym.spaces import Box, Discrete
from gym_minigrid.minigrid import MiniGridEnv, Grid, Door, Goal, Ball
from matplotlib import pyplot as plt
from numpy.linalg import norm

from ril_grid.obj import Shelf

# if true, then the reward is just the negative distance to the goal
# if false, reward is positive when distance to goal decreases
REWARD_DIST = True

# if true, make it so that the observations are in range 0-1,
# otherwise they are integers corresponding to the type of objects in view
OBS_NORMALIZED = True
# OBS_NORMALIZED = False


class RilEnv(MiniGridEnv):
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # # Drop an object
        # drop = 4
        # # Toggle/activate an object
        # toggle = 5

        # # Done completing task
        # done = 5

    def __init__(self):

        self._agent_default_pos = (5, 1)
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
        # exit tile is hardcoded atm
        self.exit = np.array((2, 1))
        super().__init__(width=6 + 2, height=7 + 2, max_steps=40)
        if not OBS_NORMALIZED:
            self.observation_space = Box(0, 7, shape=(49,), dtype=np.uint8)
        else:
            self.observation_space = Box(0, 10, shape=(49,), dtype=np.uint8)

        self.actions = RilEnv.Actions
        self.action_space = Discrete(len(self.actions))

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
        self.grid.set(*self.goal_pos, Ball())

        self.mission = "Pick up the target item and leave"

    def gen_obs(self):
        obs = super().gen_obs()
        # axis 3, layer 1 of the "image" is object type, layer 2 is "color" as single int, layer 3 is unused
        obs = obs["image"][:, :, 0].flatten()
        if OBS_NORMALIZED:
            obs = obs.astype(np.float32) / 7.0
        return obs

    def reset(self):
        self.picked_up_obj = False

        obs = super().reset()
        self.shortest_dist = norm(np.array(self.goal_pos) - np.array(self.agent_pos))
        return obs

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        just_picked_up = False

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    # add item to inventory
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    # replace item cell with empty shelf
                    self.grid.set(*fwd_pos, Shelf("purple"))
                    just_picked_up = True

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # reward calc
        if not self.picked_up_obj and not just_picked_up:
            # 1st half of trajectory, before item is in inventory
            dist = norm(np.array(self.goal_pos) - np.array(self.agent_pos))
            # print(f"goal: {self.goal_pos}, pos: {self.agent_pos}, dist: {dist}")
            if REWARD_DIST:
                reward = -dist
            else:
                reward = self.shortest_dist - dist
                if dist < self.shortest_dist:
                    self.shortest_dist = dist
        elif just_picked_up:
            # when the item has just been picked up, we change the goal calculation and give 5 points
            reward = 5
            self.picked_up_obj = True
            self.shortest_dist = norm(self.exit - np.array(self.agent_pos))
        else:
            # 2nd half of trajectory, after the item is in the inventory
            dist = norm(self.exit - np.array(self.agent_pos))
            # print(f"exit: {self.exit}, pos: {self.agent_pos}, dist: {dist}")
            if REWARD_DIST:
                reward = -dist
            else:
                reward = self.shortest_dist - dist
            if dist < 1:
                done = True
                reward = 10

        return obs, reward, done, {}
