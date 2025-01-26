import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from src import settings
from io import BytesIO
import numpy as np
from PIL import Image
import random

class Renderer:
    def __init__(self, grid_size, eligible_cells, max_response_time):
        self.grid_size = grid_size
        self.eligible_cells = eligible_cells
        self.max_response_time = max_response_time
        self.previous_target_count = 0
        self.image_dir = settings.RENDER_DIR

        # Load images once
        self.item_map = {}  # Map of item locations to image indices
        self.agent_img = mpimg.imread(os.path.join(self.image_dir, "agent.png"))
        self.target_img = mpimg.imread(os.path.join(self.image_dir, "target.png"))
        self.item_images = [
            mpimg.imread(os.path.join(self.image_dir, f"item_{i}.png")) for i in range(4)
        ]
        self.eligible_cell_img = mpimg.imread(os.path.join(self.image_dir, "eligible_cell.png"))
        self.ineligible_cell_img = mpimg.imread(os.path.join(self.image_dir, "ineligible_cell.png"))

        # Initialize figure and axis once
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        # Pre-render grid once
        self._pre_render_grid()

        self.images = []

    def _pre_render_grid(self):
        """
        Pre-render the static grid with eligible/ineligible cells, excluding the target cell.
        """
        rows, cols = self.grid_size
        self.ax.set_xlim(-0.5, cols - 0.5)
        self.ax.set_ylim(-0.5, rows - 0.5)
        self.ax.set_xticks(range(cols))
        self.ax.set_yticks(range(rows))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(False)

        for r in range(rows):
            for c in range(cols):
                if (r, c) == (2,0):  # Skip the target cell
                    continue
                img = self.eligible_cell_img if (r, c) in self.eligible_cells else self.ineligible_cell_img
                self.ax.imshow(img, extent=(c - 0.5, c + 0.5, rows - 1 - r - 0.5, rows - 1 - r + 0.5))
                
    def get_images(self):
        return np.array(self.images)

    def figure_to_numpy_array(self):
        buf = BytesIO()
        self.fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        image_array = np.transpose(image_array, (2, 0, 1))
        buf.close()
        return image_array

    def render(self, agent_loc, target_loc, item_locs, item_times, agent_load, target_count):
        """
        Render the environment dynamically (agent, target, items, sparks).
        """
        self.ax.clear()
        self._pre_render_grid()

        rows, cols = self.grid_size

        # Draw agent
        agent_x, agent_y = agent_loc
        self.ax.imshow(
            self.agent_img,
            extent=(agent_y - 0.5, agent_y + 0.5, rows - 1 - agent_x - 0.5, rows - 1 - agent_x + 0.5),
        )

        # Display the number of items carried by the agent
        if agent_load > 0:
            self.ax.text(
                agent_y, rows - 1 - agent_x + 0.3, f"{agent_load}",
                fontsize=10, ha="center", color="white", bbox=dict(facecolor="black", alpha=0.5),
            )

        # Draw the target cell background
        target_x, target_y = target_loc
        self.ax.imshow(
            self.target_img,
            extent=(target_y - 0.5, target_y + 0.5, rows - 1 - target_x - 0.5, rows - 1 - target_x + 0.5),
        )

        # Draw the target count, centered in the cell
        self.ax.text(
            target_y, rows - 1 - target_x + 0.1, f"{target_count}",
            fontsize=12, ha="center", color="red", bbox=dict(facecolor="white", alpha=0.7),
        )

        # Draw items
        for (item_x, item_y), response_time in zip(item_locs, item_times):
            # Assign a random ID/image index if this is a new item
            if (item_x, item_y) not in self.item_map:
                self.item_map[(item_x, item_y)] = random.choice(range(4))

            image_index = self.item_map[(item_x, item_y)]
            item_image = self.item_images[image_index]

            # Calculate item size based on remaining lifetime
            lifetime = (self.max_response_time - response_time) / self.max_response_time
            size_factor = 0.3 + 0.7 * lifetime
            self.ax.imshow(
                item_image,
                extent=(item_y - size_factor / 2, item_y + size_factor / 2,
                        rows - 1 - item_x - size_factor / 2, rows - 1 - item_x + size_factor / 2),
            )

        # Show sparks if the target count increased
        '''if target_count > self.previous_target_count:
            self.ax.imshow(
                self.sparks_img,
                extent=(target_y - 0.5, target_y + 0.5, rows - 1 - target_x - 0.5, rows - 1 - target_x + 0.5),
            )'''

        # Update the previous target count
        self.previous_target_count = target_count

        # Store image for logging
        image_array = self.figure_to_numpy_array()
        self.images.append(image_array)

        plt.pause(0.0001)



