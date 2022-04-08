from gym_minigrid.minigrid import Box, COLORS
from gym_minigrid.rendering import point_in_rect, fill_coords


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
