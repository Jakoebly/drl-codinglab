import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from src import settings
from io import BytesIO
import numpy as np
from PIL import Image



class Renderer:
    def __init__(self, grid_size, eligible_cells, max_response_time, mode):
        self.grid_size = grid_size
        self.eligible_cells = eligible_cells
        self.max_response_time = max_response_time
        self.previous_target_count = 0
        self.image_dir = settings.RENDER_DIR

        # Load images once
        self.agent_img = mpimg.imread(os.path.join(self.image_dir, "agent.png"))
        self.target_img = mpimg.imread(os.path.join(self.image_dir, "target.png"))
        self.item_img = mpimg.imread(os.path.join(self.image_dir, "item_0.png"))
        self.sparks_img = mpimg.imread(os.path.join(self.image_dir, "sparks.png"))
        self.eligible_cell_img = mpimg.imread(os.path.join(self.image_dir, "eligible_cell.png"))
        self.ineligible_cell_img = mpimg.imread(os.path.join(self.image_dir, "ineligible_cell.png"))

        # Initialize figure and axis once
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        # Pre-render grid once
        self._pre_render_grid()

        self.train_images = []
        self.val_images = []
        self.mode = mode

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

        for r in range(rows):
            for c in range(cols):
                img = self.eligible_cell_img if (r, c) in self.eligible_cells else self.ineligible_cell_img
                self.ax.imshow(img, extent=(c - 0.5, c + 0.5, rows - 1 - r - 0.5, rows - 1 - r + 0.5))

    def get_images(self):
        return np.array(self.train_images), np.array(self.val_images)

    def figure_to_numpy_array(self):
        buf = BytesIO()
        self.fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        image_array = np.transpose(image_array, (2, 0, 1))
        buf.close()
        return image_array

    def render(self, agent_loc, target_loc, item_locs, item_times, agent_load, target_count, show_render=False):
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

        # If the agent is carrying an item, display it in its left hand
        if agent_load > 0:
            item_size = 0.2  # Adjust the size of the item
            self.ax.imshow(
                self.apples_one_img,
                extent=(agent_y - 0.4, agent_y - 0.4 + item_size,  # Positioned slightly left of the agent
                        rows - 1 - agent_x + 0, rows - 1 - agent_x + 0 + item_size),
                # Positioned slightly above the agent
            )

        self.ax.text(
            agent_y, rows - 1 - agent_x + 0.3, f"Load: {agent_load}",
            fontsize=10, ha="center", color="white", bbox=dict(facecolor="black", alpha=0.5),
        )

        # Select the appropriate target image based on the target count
        overlay_img = None
        if target_count == 1:
            overlay_img = self.apples_one_img
        elif 3 <= target_count <= 6:
            overlay_img = self.apples_few_img
        elif target_count > 6:
            overlay_img = self.apples_lot_img

        # Draw the target cell background
        target_x, target_y = target_loc
        self.ax.imshow(
            self.target_img,
            extent=(target_y - 0.5, target_y + 0.5, rows - 1 - target_x - 0.5, rows - 1 - target_x + 0.5),
        )

        # Draw the overlay image anchored at the bottom of the cell if applicable
        if overlay_img is not None:
            image_size = 0.3  # Adjust image size
            self.ax.imshow(
                overlay_img,
                extent=(target_y - 0.5 * image_size, target_y + 0.5 * image_size,  # Horizontal scaling
                        rows - 1 - target_x - 0.5, rows - 1 - target_x - 0.5 + image_size),  # Anchor at bottom
            )

        # Draw the target count, centered in the cell
        self.ax.text(
            target_y, rows - 1 - target_x + 0.1, f"{target_count}",
            fontsize=12, ha="center", color="red", bbox=dict(facecolor="white", alpha=0.7),
        )

        # Draw items
        for (item_x, item_y), response_time in zip(item_locs, item_times):
            lifetime = (self.max_response_time - response_time) / self.max_response_time
            size_factor = 0.3 + 0.7 * lifetime
            self.ax.imshow(
                self.item_img,
                extent=(item_y - size_factor / 2, item_y + size_factor / 2,
                        rows - 1 - item_x - size_factor / 2, rows - 1 - item_x + size_factor / 2),
            )

        # Show sparks if the target count increased
        if target_count > self.previous_target_count:
            self.ax.imshow(
                self.sparks_img,
                extent=(target_y - 0.5, target_y + 0.5, rows - 1 - target_x - 0.5, rows - 1 - target_x + 0.5),
            )

        # Update the previous target count
        self.previous_target_count = target_count

        # Store image for logging
        image_array = self.figure_to_numpy_array()
        if self.mode == "training":
            self.train_images.append(image_array)
        else:
            self.val_images.append(image_array)

