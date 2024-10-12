import os
import textwrap
import cv2
import easyocr.recognition
import numpy as np
import easyocr
from typing import List


class BubblePainter:

    def __init__(self, lang_list=["en"]) -> None:
        # Initialize the EasyOCR reader
        self.reader = easyocr.Reader(lang_list)

    # Function to group text regions that are close together
    def group_text_boxes(self, boxes, touch_threshold=5):
        """
        Groups text boxes based on the distance from the combined bounding box edges.
        """
        grouped_boxes = []
        visited = np.zeros(len(boxes), dtype=bool)  # Track visited boxes

        for i in range(len(boxes)):
            if visited[i]:  # Skip already visited boxes
                continue

            current_group = [boxes[i]]  # Start a new group with the current box
            visited[i] = True  # Mark as visited

            # Calculate the combined bounding box for the current group
            min_x = boxes[i][0][0]
            min_y = boxes[i][0][1]
            max_x = boxes[i][1][0]
            max_y = boxes[i][1][1]

            # Check against all other boxes
            for j in range(len(boxes)):
                if visited[j]:  # Skip already visited boxes
                    continue

                # Get the coordinates of the current box
                box_top_left, box_bottom_right = boxes[j]

                # Calculate new edges
                new_min_x = box_top_left[0]
                new_min_y = box_top_left[1]
                new_max_x = box_bottom_right[0]
                new_max_y = box_bottom_right[1]

                # Check if the new box is close to the combined bounding box
                if (
                    new_min_x <= max_x + touch_threshold
                    and new_max_x >= min_x - touch_threshold
                    and new_min_y <= max_y + touch_threshold
                    and new_max_y >= min_y - touch_threshold
                ):

                    # Update combined bounding box
                    current_group.append(boxes[j])
                    visited[j] = True  # Mark as visited

                    # Update the combined bounding box edges
                    min_x = min(min_x, new_min_x)
                    min_y = min(min_y, new_min_y)
                    max_x = max(max_x, new_max_x)
                    max_y = max(max_y, new_max_y)

            grouped_boxes.append(current_group)  # Add the current group to the results

        return grouped_boxes

    # Function to draw outlined text
    def draw_outlined_text(
        self,
        img,
        text,
        position,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1,
        color=(255, 255, 255),
        thickness=2,
        outline_thickness=2,
    ):
        # Draw the text with outline
        x, y = position
        # Draw outline by putting text multiple times with offset
        for dx in range(-outline_thickness, outline_thickness + 1):
            for dy in range(-outline_thickness, outline_thickness + 1):
                if dx != 0 or dy != 0:  # Avoid drawing in the center
                    cv2.putText(
                        img,
                        text,
                        (x + dx, y + dy),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                    )

        # Draw the main text over the outline
        cv2.putText(img, text, position, font, font_scale, color, thickness)

    # Function to get the appropriate font size for the entire image (based on the smallest bubble)
    def calculate_max_font_size_for_bubbles(
        self,
        text,
        bubbles,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        min_font_scale=0.5,
        max_font_scale=2.0,
        thickness=1,
    ):
        smallest_box_width = min([box[1][0] - box[0][0] for box in bubbles])

        # Find the maximum font size that fits in the smallest bubble
        for font_scale in np.arange(max_font_scale, min_font_scale, -0.1):
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            if text_size[0] <= smallest_box_width:
                return font_scale

        return min_font_scale  # Fallback in case none fit

    # Function to draw new text inside the inpainted bubble with consistent font size
    def draw_text_on_bubble(
        self,
        img,
        text,
        box,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        color=(0, 0, 0),
        thickness=1,
        font_scale=1.0,
    ):
        top_left = box[0]
        bottom_right = box[1]

        # Get the bounding box for the bubble
        x, y = top_left
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]

        # Split the text into words for wrapping
        words = text.split(" ")
        lines = []
        current_line = ""

        # Wrap text based on bubble width
        while words:
            word = words.pop(0)
            test_line = f"{current_line} {word}".strip()
            text_size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)

            if text_size[0] <= width:  # If it fits in width
                current_line = test_line  # Update the line
            else:
                lines.append(current_line)  # Push the line and start new
                current_line = word  # Start a new line with the word

        if current_line:
            lines.append(current_line)  # Add the last line

        # Calculate the total height of the text
        total_text_height = sum(
            [cv2.getTextSize(line, font, font_scale, thickness)[1] for line in lines]
        )

        # If total text height exceeds bubble height, decrease the font size
        while total_text_height > height and font_scale > 0.5:
            font_scale -= 0.1
            total_text_height = sum(
                [
                    cv2.getTextSize(line, font, font_scale, thickness)[1]
                    for line in lines
                ]
            )

        # Calculate starting y position for centering the text
        y_offset = y + (height - total_text_height) // 2

        # Write each line within the bubble region, centered
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            x_centered = x + (width - text_size[0]) // 2  # Center the text horizontally
            self.draw_outlined_text(
                img, line, (x_centered, y_offset), font, font_scale, color, thickness
            )
            y_offset += text_size[1] + 5  # Move to the next line

    # Function to create a mask with ellipses for inpainting
    def create_inpaint_mask_with_ellipse(self, img_shape, boxes, margin=-10):
        mask = np.zeros(img_shape[:2], dtype="uint8")

        for top_left, bottom_right in boxes:
            # Calculate the center of the bounding box
            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2

            # Calculate the axes lengths (width and height)
            axes_x = (bottom_right[0] - top_left[0]) // 2 - margin
            axes_y = (bottom_right[1] - top_left[1]) // 2 - margin

            # Ensure the axes are non-negative
            axes_x = max(axes_x, 1)
            axes_y = max(axes_y, 1)

            # Draw the ellipse on the mask
            cv2.ellipse(
                mask,
                (center_x, center_y),  # Center of the ellipse
                (axes_x, axes_y),  # Axes lengths
                0,  # Angle of rotation (0 means horizontal ellipse)
                0,
                360,  # Start and end angle (full ellipse)
                (255, 255, 255),  # Color (white for mask)
                -1,  # Thickness (-1 fills the ellipse)
            )

        return mask

    # Main function that processes multiple images and replaces text in bubbles
    def inpaint_text_with_new_text(
        self,
        img_paths: List[str],
        texts: List[str],
        draw_bounds=False,
        edge_threshold=10,
    ) -> List[np.ndarray]:
        processed_images = []

        for img_path, new_text in zip(img_paths, texts):
            img = cv2.imread(img_path)

            # Use EasyOCR to read the text and get bounding boxes
            results = self.reader.readtext(img)

            # Extract the bounding boxes
            boxes = []
            for box in results:
                if len(box) > 0 and len(box[0]) == 4:  # Ensure we have 4 corners
                    top_left = (int(box[0][0][0]), int(box[0][0][1]))  # Top-left corner
                    bottom_right = (
                        int(box[0][2][0]),
                        int(box[0][2][1]),
                    )  # Bottom-right corner
                    boxes.append((top_left, bottom_right))

            # Group text boxes that are close together
            grouped_boxes = self.group_text_boxes(boxes)

            # Filter groups: Ignore very close text bubbles with 2 or fewer sentences
            filtered_groups = []
            for group in grouped_boxes:
                if len(group) >= 2:  # Ignore groups with less than 2 boxes
                    top_left_x = int(min([box[0][0] for box in group]))
                    top_left_y = int(min([box[0][1] for box in group]))
                    bottom_right_x = int(max([box[1][0] for box in group]))
                    bottom_right_y = int(max([box[1][1] for box in group]))

                    # Check if the group is too close to the edge
                    if (
                        top_left_x > edge_threshold
                        and top_left_y > edge_threshold
                        and bottom_right_x < img.shape[1] - edge_threshold
                        and bottom_right_y < img.shape[0] - edge_threshold
                    ):
                        filtered_groups.append(
                            ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
                        )

            # Sort the bubbles from top to bottom based on y-coordinate
            filtered_groups.sort(key=lambda box: box[0][1])

            # Create a mask with ellipses for inpainting
            mask = self.create_inpaint_mask_with_ellipse(img.shape, filtered_groups)

            # Inpaint the image to remove the original text in the bubble
            inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

            # Calculate a uniform font size based on the smallest bubble
            if filtered_groups:
                max_font_size = self.calculate_max_font_size_for_bubbles(
                    new_text, filtered_groups
                )

                # Distribute text across bubbles and draw it
                new_lines = new_text.split("\n")
                bubbles_count = len(filtered_groups)
                lines_per_bubble = len(new_lines) // bubbles_count
                extra_lines = len(new_lines) % bubbles_count
                start_index = 0

                for i, (top_left, bottom_right) in enumerate(filtered_groups):
                    num_lines = lines_per_bubble + (1 if i < extra_lines else 0)
                    bubble_text = " ".join(
                        new_lines[start_index : start_index + num_lines]
                    )
                    start_index += num_lines

                    # Draw text inside the bubble
                    self.draw_text_on_bubble(
                        inpainted_img,
                        bubble_text,
                        (top_left, bottom_right),
                        font_scale=max_font_size,
                    )

            processed_images.append(inpainted_img)

        return processed_images


if __name__ == "__main__":
    bubble_painter = BubblePainter()

    filenames = [
        "images/1-1.png",
        "images/1-2.png",
        "images/1-3.png",
        "images/1-4.png",
        "images/2-1.png",
        "images/2-2.png",
        "images/2-3.png",
    ]

    # Process the images (inpaint and draw new text)
    processed_images = bubble_painter.inpaint_text_with_new_text(
        filenames,
        [
            f"Here is different text for image {i + 1}.\nIt can also span multiple lines."
            for i, _ in enumerate(filenames)
        ],
    )

    # Display and save the results
    for img, filename in zip(processed_images, filenames):
        result_path = f"./images/result-{os.path.basename(filename)}"
        cv2.imwrite(result_path, img)
