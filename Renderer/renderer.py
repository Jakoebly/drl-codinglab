import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


class Renderer:
    def __init__(self, grid_size, eligible_cells, max_response_time):
        """
        Initialize the renderer.
        :param grid_size: Tuple (rows, cols) representing the grid dimensions.
        :param eligible_cells: List of tuples [(row, col), ...] for eligible cells.
        :param max_response_time: Maximum response time for items (used to calculate lifetimes).
        """
        self.grid_size = grid_size
        self.eligible_cells = eligible_cells
        self.max_response_time = max_response_time
        self.previous_target_count = 0
        self.image_dir = "drl-codinglab\Renderer\Images"

        # Load images
        self.agent_img = mpimg.imread(os.path.join(self.image_dir, "agent.png"))
        self.target_img = mpimg.imread(os.path.join(self.image_dir, "target.png"))
        self.item_img = mpimg.imread(os.path.join(self.image_dir, "item_0.png"))  # Single item image
        self.sparks_img = mpimg.imread(os.path.join(self.image_dir, "sparks.png"))
        self.eligible_cell_img = mpimg.imread(os.path.join(self.image_dir, "eligible_cell.png"))
        self.ineligible_cell_img = mpimg.imread(os.path.join(self.image_dir, "ineligible_cell.png"))

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self._pre_render_grid()

    def _pre_render_grid(self):
        """
        Pre-render the static grid with eligible/ineligible cells.
        """
        rows, cols = self.grid_size
        self.ax.set_xlim(-0.5, cols - 0.5)
        self.ax.set_ylim(-0.5, rows - 0.5)
        self.ax.set_xticks(range(cols))
        self.ax.set_yticks(range(rows))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(False)

        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        for cell in all_cells:
            x, y = cell
            img = self.eligible_cell_img if cell in self.eligible_cells else self.ineligible_cell_img
            self.ax.imshow(img, extent=(y - 0.5, y + 0.5,
                                        rows - 1 - x - 0.5, rows - 1 - x + 0.5))

    def render(self, agent_loc, target_loc, item_locs, item_times, agent_load, target_count):
        """
        Render the environment dynamically (agent, target, items, sparks).
        :param agent_loc: Tuple (row, col) for the agent's position.
        :param target_loc: Tuple (row, col) for the target's position.
        :param item_locs: List of tuples [(row, col), ...] for item positions.
        :param item_times: List of integers representing the response time of each item.
        :param agent_load: Integer indicating the number of items the agent is carrying.
        :param target_count: Integer indicating the number of items dropped at the target.
        """
        self.ax.clear()
        self._pre_render_grid()

        rows, cols = self.grid_size

        agent_x, agent_y = agent_loc
        self.ax.imshow(self.agent_img, extent=(agent_y - 0.5, agent_y + 0.5,
                                               rows - 1 - agent_x - 0.5, rows - 1 - agent_x + 0.5))
    
        self.ax.text(agent_y - 0.3, rows - 1 - agent_x, f"{agent_load}",
                     color="black", fontsize=12, ha="left", va="center",
                     bbox=dict(facecolor="white", edgecolor="black"))

        target_x, target_y = target_loc
        self.ax.imshow(self.target_img, extent=(target_y - 0.5, target_y + 0.5,
                                                rows - 1 - target_x - 0.5, rows - 1 - target_x + 0.5))
   
        self.ax.text(target_y, rows - 1 - target_x, f"{target_count}",
                     color="red", fontsize=12, ha="center", va="center",
                     bbox=dict(facecolor="white", edgecolor="black"))

        if target_count > self.previous_target_count:
            self.ax.imshow(self.sparks_img, extent=(target_y - 0.5, target_y + 0.5,
                                                    rows - 1 - target_x - 0.5, rows - 1 - target_x + 0.5))

        for (item_x, item_y), response_time in zip(item_locs, item_times):
            lifetime = (self.max_response_time - response_time) / self.max_response_time
            size_factor = 0.3 + 0.7 * lifetime  # Scale size from 0.3 to 1.0
            self.ax.imshow(self.item_img, extent=(item_y - 0.5 * size_factor, item_y + 0.5 * size_factor,
                                                  rows - 1 - item_x - 0.5 * size_factor, rows - 1 - item_x + 0.5 * size_factor))

        self.previous_target_count = target_count

        plt.pause(0.1)
