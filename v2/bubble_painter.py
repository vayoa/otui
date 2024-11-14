import os
import re
import textwrap
import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image
import nodes


# Function to find bounding boxes of each bubble in the mask
def find_bubble_bounding_boxes(
    mask: np.ndarray,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(((x, y), (x + w, y + h)))

    return boxes


# Function to draw outlined text
def draw_outlined_text(
    img,
    text,
    position,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1,
    color=(0, 0, 0),
    thickness=2,
    outline_thickness=2,
):
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


# Calculate font size for bubbles
def calculate_max_font_size_for_bubbles(
    text,
    bubbles,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    min_font_scale=0.5,
    max_font_scale=2.0,
    thickness=1,
):
    smallest_box_width = min([box[1][0] - box[0][0] for box in bubbles])

    for font_scale in np.arange(max_font_scale, min_font_scale, -0.1):
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        if text_size[0] <= smallest_box_width:
            return font_scale

    return min_font_scale


# Draw text inside the bubble
def draw_text_on_bubble(
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

    x, y = top_left
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    # Split text into lines based on bubble width
    words = text.split(" ")
    lines = []
    current_line = ""

    while words:
        word = words.pop(0)
        test_line = f"{current_line} {word}".strip()
        text_size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if text_size[0] <= width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Calculate total text height
    total_text_height = sum(
        [cv2.getTextSize(line, font, font_scale, thickness)[1] for line in lines]
    )

    while total_text_height > height and font_scale > 0.5:
        font_scale -= 0.1
        total_text_height = sum(
            [cv2.getTextSize(line, font, font_scale, thickness)[1] for line in lines]
        )

    # Center text vertically in bubble
    y_offset = y + (height - total_text_height) // 2

    # Write each line
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        x_centered = x + (width - text_size[0]) // 2
        draw_outlined_text(
            img, line, (x_centered, y_offset), font, font_scale, color, thickness
        )
        y_offset += text_size[1] + 5


# Helper function to split text on newlines, periods, or commas
def split_text_for_bubbles(text: str, max_bubbles: int) -> List[str]:
    # Split on newlines first
    split_text = text.splitlines()
    if len(split_text) >= max_bubbles:
        return split_text

    # If there are fewer lines than bubbles, split further on periods and commas
    further_split = re.split(r"[.,]", text)
    further_split = [line.strip() for line in further_split if line.strip()]

    # Distribute text to match the number of bubbles
    if len(further_split) >= max_bubbles:
        return further_split

    # Otherwise, group remaining lines if they are still fewer than bubbles
    grouped_text = []
    line_buffer = ""
    for line in further_split:
        if len(grouped_text) + 1 < max_bubbles:
            grouped_text.append(line)
        else:
            # Combine the remaining lines if less than bubbles are available
            line_buffer += f"{line} "
    if line_buffer:
        grouped_text.append(line_buffer.strip())

    return grouped_text


# Main function to process images with empty bubbles and masks
def add_text_to_bubbles(
    empty_images: list[np.ndarray] | list[str],
    masks: list[np.ndarray] | list[str],
    texts: List[str],
    edge_threshold=10,
) -> List[np.ndarray]:
    processed_images = []

    for empty_image, mask, new_text in zip(empty_images, masks, texts):
        if isinstance(empty_image, str):
            img = cv2.imread(empty_image)
        else:
            img = cv2.cvtColor(np.array(empty_image), cv2.COLOR_RGB2BGR)

        if isinstance(mask, str):
            mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        else:
            mask_img = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY)

        # Find bounding boxes for bubbles in the mask
        bubble_boxes = find_bubble_bounding_boxes(mask_img)

        # Filter boxes not near the edges
        filtered_boxes = [
            box
            for box in bubble_boxes
            if box[0][0] > edge_threshold
            and box[0][1] > edge_threshold
            and box[1][0] < img.shape[1] - edge_threshold
            and box[1][1] < img.shape[0] - edge_threshold
        ]

        # Sort bubbles by y-coordinate
        filtered_boxes.sort(key=lambda box: box[0][1])

        # Calculate uniform font size based on the smallest bubble
        if filtered_boxes:
            max_font_size = calculate_max_font_size_for_bubbles(
                new_text, filtered_boxes
            )

            # Split text across bubbles
            new_lines = split_text_for_bubbles(new_text, len(filtered_boxes))
            bubbles_count = len(filtered_boxes)
            lines_per_bubble = len(new_lines) // bubbles_count
            extra_lines = len(new_lines) % bubbles_count
            start_index = 0

            for i, box in enumerate(filtered_boxes):
                num_lines = lines_per_bubble + (1 if i < extra_lines else 0)
                bubble_text = " ".join(new_lines[start_index : start_index + num_lines])
                start_index += num_lines

                # Draw text inside the bubble
                draw_text_on_bubble(img, bubble_text, box, font_scale=max_font_size)

        processed_images.append(img)

    return processed_images


def clear_bubbles_workflow(image_node_output: nodes.IMAGE):
    sam = nodes.SamModelLoader()
    dino = nodes.GroundingDinoModelLoader()
    segment = nodes.GroundingDinoSAMSegment(
        sam_model=sam.outputs["SAM_MODEL"],
        grounding_dino_model=dino.outputs["GROUNDING_DINO_MODEL"],
        image=image_node_output,
        prompt="white comic text bubble with text",
        threshold=0.3,
    )
    invert = nodes.InvertMask(segment.outputs["MASK"])
    convert_mask = nodes.MaskToImage(segment.outputs["MASK"])
    composite = nodes.ImageCompositeMasked(
        destination=convert_mask.outputs["IMAGE"],
        source=image_node_output,
        mask=invert.outputs["MASK"],
    )
    workflow_nodes = [sam, dino, segment, invert, convert_mask, composite]
    return nodes.workflow(
        *workflow_nodes,
        nodes.SaveImageWebsocket(images=convert_mask.outputs["IMAGE"]).rename(
            "bubbles"
        ),
        nodes.SaveImageWebsocket(images=composite.outputs["IMAGE"]).rename("final"),
    )


def add_text(results, text) -> Image.Image:
    results = {result: results[result][-1] for result in results}
    mask = cv2.cvtColor(np.array(results["bubbles_OUTPUT"]), cv2.COLOR_RGB2BGR)
    empty_image = cv2.cvtColor(np.array(results["final_OUTPUT"]), cv2.COLOR_RGB2BGR)
    return Image.fromarray(add_text_to_bubbles([empty_image], [mask], [text])[-1])


if __name__ == "__main__":
    from eyes import Eyes

    def clear_bubbles(eyes: Eyes, img: Image.Image):
        load_image_node = nodes.ETN_LoadImageBase64(image=img)
        return list(
            eyes.get_images(
                {**load_image_node.json()}
                | clear_bubbles_workflow(load_image_node.outputs["IMAGE"])
            )
        )

    def add_dialog(
        eyes: Eyes, images: list[Image.Image] | Image.Image, texts: list[str]
    ) -> list[Image.Image]:
        if not isinstance(images, list):
            images = [images]

        if not isinstance(texts, list):
            texts = [texts]

        empty_images, masks = [], []

        for image in images:
            results = clear_bubbles(eyes, image)[-1]
            if results is not None:
                results = {result: results[result][-1] for result in results}
                masks.append(
                    cv2.cvtColor(np.array(results["bubbles_OUTPUT"]), cv2.COLOR_RGB2BGR)
                )
                empty_images.append(
                    cv2.cvtColor(np.array(results["final_OUTPUT"]), cv2.COLOR_RGB2BGR)
                )

        # Process images and add text to bubbles
        return [
            Image.fromarray(image)
            for image in add_text_to_bubbles(empty_images, masks, texts)
        ]

    eyes = Eyes()
    images = [
        "./images/image1.png",
        "./images/image1.png",
        "./images/image2.png",
    ]
    processed_images = add_dialog(
        eyes,
        [Image.open(url) for url in images],
        [
            "Text for the first image bubble.\nThis is some longggggg text\nMake sure you have it all",
            "Hey, this is another test!",
            "Hello!",
        ],
    )

    # Save or display the results
    for i, (img, filename) in enumerate(zip(processed_images, images)):
        result_path = f"./images/result-{os.path.basename(filename)}-{i}.png"
        img.save(result_path)
