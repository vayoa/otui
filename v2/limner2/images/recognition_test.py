import cv2
import easyocr

# Create an EasyOCR reader instance
reader = easyocr.Reader(["en"])

# Path to the image
img_path = "images/2-1.png"

# Read the image using OpenCV
img = cv2.imread(img_path)

# Use EasyOCR to read the text and get bounding boxes
results = reader.readtext(img)

# Iterate over the results and draw bounding boxes
for bbox, text, prob in results:
    # Unpack the bounding box
    top_left = tuple(map(int, bbox[0]))  # Top-left corner
    bottom_right = tuple(map(int, bbox[2]))  # Bottom-right corner

    # Draw the rectangle on the image
    cv2.rectangle(
        img, top_left, bottom_right, (0, 255, 0), 2
    )  # Green color, thickness 2

    # Optionally, put the text on the image
    cv2.putText(
        img,
        text,
        (top_left[0], top_left[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

# Optionally save the image with bounding boxes
output_path = "images/1-2_bboxes.png"
cv2.imwrite(output_path, img)
