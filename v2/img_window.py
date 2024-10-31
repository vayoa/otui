from win32mica import ApplyMica, MicaTheme, MicaStyle
import sys
import threading
import time
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PIL import Image, ImageQt, ImageFilter, ImageDraw
from eyes import Eyes


# GUI signal handler
class ImageUpdater(QObject):
    update_image_signal = Signal(QPixmap)  # Change signal to use QPixmap

    def __init__(self, eyes: Eyes):
        super().__init__()
        self.eyes = eyes
        self.update_image_signal.connect(self.update_image)
        self.window = None
        self.preview_thread = None
        self.stop_event = (
            threading.Event()
        )  # Event to control the stopping of the thread

    def start_gui(self):
        # Initialize GUI on the main thread
        app = QApplication.instance() or QApplication(sys.argv)
        self.window = ImagePreviewWindow(
            self, app.primaryScreen().size().toTuple()
        )  # Initialize the window here
        self.window.show()  # Show the window immediately
        app.exec()  # This starts the Qt event loop

    def update_image(self, pixmap: QPixmap):
        if self.window:
            self.window.set_image(pixmap)

    def preview(
        self,
        positive,
        negative=None,
        dimensions=None,
        character_image=None,
        lcm=None,
        checkpoint=None,
        steps=None,
        sampler_name=None,
        cfg=None,
    ):
        # If the window is hidden, show it
        if self.window.isHidden():
            self.window.show()

        # Clear the image immediately to prevent flash of previous images
        self.window.clear_image()  # Clear the image on the window

        # Signal the previous thread to stop
        if self.preview_thread and self.preview_thread.is_alive():
            self.stop_event.set()  # Signal the thread to stop
            self.preview_thread.join()  # Wait for it to finish

        # Reset the stop event for the new thread
        self.stop_event.clear()

        # Start a new preview thread
        self.preview_thread = threading.Thread(
            target=lambda: self.generate_images(
                positive,
                negative=negative,
                dimensions=dimensions,
                character_image=character_image,
                lcm=lcm,
                checkpoint=checkpoint,
                steps=steps,
                sampler_name=sampler_name,
                cfg=cfg,
            ),
            daemon=True,
        )
        self.preview_thread.start()

    def feather_edges(self, image, fade_margin=15):
        # Load the image and ensure it has an alpha channel for transparency
        image = image.convert("RGBA")

        # Create a mask with a transparent center and solid edges
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Draw a filled rectangle in the center with fade_margin from the edges
        draw.rectangle(
            (
                fade_margin,
                fade_margin,
                image.width - fade_margin,
                image.height - fade_margin,
            ),
            fill=255,
        )

        # Blur the mask to create a gradient fade on the rectangle edges
        mask = mask.filter(ImageFilter.GaussianBlur(fade_margin / 2))

        # Apply the mask to the image's alpha channel to make edges transparent
        transparent_image = image.copy()
        transparent_image.putalpha(mask)

        return transparent_image

    def generate_images(
        self,
        positive,
        negative=None,
        dimensions=None,
        character_image=None,
        lcm=None,
        checkpoint=None,
        steps=None,
        sampler_name=None,
        cfg=None,
    ):
        for i, (_, previews) in enumerate(
            self.eyes.generate_yield(
                positive,
                negative=negative,
                dimensions=dimensions,
                character_image=character_image,
                lcm=lcm,
                checkpoint=checkpoint,
                steps=steps,
                sampler_name=sampler_name,
                cfg=cfg,
            )
        ):  # Simulate 3 preview images
            if self.stop_event.is_set():  # Check if we should stop
                return  # Exit the thread if stopping

            if previews is not None:
                keys = previews.keys()
                keys = tuple(filter(lambda key: "SaveImageWebsocket" in key, keys))
                # get the final image if ready, otherwise get previews
                preview_image = (
                    self.feather_edges(previews[keys[-1]][-1])
                    if keys
                    else self.feather_edges(
                        previews[list(previews.keys())[0]][-1],
                        105 - (((105 - 35) // 24) * (i // 2)),
                    )
                )

                # Convert the PIL image to QPixmap
                qt_image = ImageQt.ImageQt(preview_image)
                pixmap = QPixmap.fromImage(qt_image)

                # Emit the signal to update the image in the GUI
                self.update_image(pixmap)  # Directly call the update function

                if keys:
                    break


# GUI window class
class ImagePreviewWindow(QMainWindow):
    def __init__(self, image_updater: ImageUpdater, screen_size):
        super().__init__()
        self.image_updater = image_updater
        self.setAttribute(Qt.WA_TranslucentBackground)  # type: ignore
        self.setWindowTitle("Image Previews")
        w, h = 576, 448
        pad = 20
        self.setGeometry(screen_size[0] - w - pad, 2 * pad, w, h)

        ApplyMica(
            self.winId(),
            MicaTheme.AUTO,  # type: ignore
            MicaStyle.DEFAULT,  # type: ignore
        )

        # Label to display the image
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.image_label.setMinimumSize(100, 100)  # Set a minimum size for the label

        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.image = None

    def set_image(self, pixmap: QPixmap):
        self.image = pixmap
        # Scale the image to fit the label's current size while maintaining aspect ratio
        scaled_pixmap = self.scale_image(pixmap, self.image_label.size())
        self.image_label.setPixmap(scaled_pixmap)

    def scale_image(self, pixmap: QPixmap, target_size):
        # Get original dimensions
        original_width = pixmap.width()
        original_height = pixmap.height()

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height

        # Get the dimensions of the target size
        target_width, target_height = target_size.width(), target_size.height()

        # Calculate new dimensions while maintaining the aspect ratio
        if target_width / target_height > aspect_ratio:
            new_width = target_height * aspect_ratio
            new_height = target_height
        else:
            new_width = target_width
            new_height = target_width / aspect_ratio

        # Create a new pixmap with the scaled dimensions
        return pixmap.scaled(
            new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def clear_image(self):
        # Clear the image immediately when a new preview starts
        self.image_label.clear()  # Clear the QLabel content
        self.image_label.setPixmap(
            QPixmap()
        )  # Set an empty pixmap to avoid flashing the old image

    def closeEvent(self, event):
        # When the window is closed, we want to ensure it can be reopened.
        event.ignore()  # Ignore the close event
        self.hide()  # Hide the window instead

    def resizeEvent(self, event):
        # Ensure image scales with window resizing
        if self.image:
            self.set_image(self.image)
        super().resizeEvent(event)


def main():
    # Initialize ImageUpdater for handling GUI updates
    image_updater = ImageUpdater()

    # Start GUI in a separate thread
    gui_thread = threading.Thread(target=image_updater.start_gui, daemon=True)
    gui_thread.start()

    # Allow some time for the GUI to initialize before sending previews
    time.sleep(1)
    while True:
        command = input("Press Enter to generate a preview (or type 'exit' to quit): ")
        if command.lower() == "exit":
            break
        image_updater.preview()  # Start a new preview


if __name__ == "__main__":
    main()
