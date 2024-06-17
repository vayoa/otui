from PIL import Image
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Markdown, TextArea, Static
from rich_pixels import Pixels
from textual import events


class PixelsContainer(Container):
    def __init__(self, img, *args, **kwargs):
        self.img = img
        super().__init__(*args, **kwargs)

    def render(self):
        size = self.size
        return Pixels.from_image(self.img, resize=(size[0], size[1] * 2))


class UserInput(TextArea):
    BINDINGS = [
        # *TextArea.BINDINGS,
        ("ctrl+e", "submit_input", "Submit Input"),
    ]

    def action_submit_input(self):
        assert False


class Terminal(Static):
    def compose(self):
        yield Markdown(
            "# Hey there\n## What's going on?\n> **hope everything is** *good*"
        )
        yield UserInput()


class OtuiGame(App):
    """A Textual app to manage stopwatches."""

    BINDINGS = [
        ("ctrl+t", "toggle_terminal", "Toggle Terminal"),
    ]

    CSS_PATH = "./textual.css"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Container(
            PixelsContainer(Image.open("./image.png")),
            Terminal(),
        )
        yield Header(
            show_clock=True,
        )
        yield Footer()

    def action_toggle_terminal(self) -> None:
        terminal = self.query_one(Terminal)
        self.set_focus(None)
        if terminal.has_class("-hidden"):
            terminal.remove_class("-hidden")
            self.query_one(UserInput).focus()
        else:
            if terminal.query("*:focus"):
                self.screen.set_focus(None)
            terminal.add_class("-hidden")


if __name__ == "__main__":
    OtuiGame().run()
