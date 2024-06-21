import argparse
import tempfile
from threading import Thread
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.live import Live
from rich.columns import Columns
from rich.panel import Panel
from rich.rule import Rule
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from langchain_community.chat_models import ChatOllama
from brain import Brain, JSONFormatter


class CommandCompleter(Completer):

    def __init__(self, characters):
        characters = {
            name: (
                character["prompt"][:20] + "..."
                if len(character["prompt"]) > 20
                else character["prompt"]
            )
            for name, character in characters.items()
        }

        self.commands = {
            "bye | quit | exit": "quit",
            "hijack": "hijack",
            "auto-hijack | ah": "toggle auto-hijack mode",
            "show | s": "generates a picture",
            "auto-show | as": "toggle auto-show mode (generates a picture for every response)",
            "messages | m": "shows the current message history",
            "characters | chars": {
                "meta": "shows all seen characters or describes a specific one",
                "commands": characters,
            },
        }

        self.characters = ["hey", "hello honey"]

    def complete(self, text, commands, prefix=""):
        if prefix:
            user_commands = text.split(prefix)
        else:
            user_commands = [text]
        if (not prefix or prefix in text) and user_commands:
            last_user_command = user_commands[-1]

            for command in commands:
                shown_alias = None
                aliases = [a.strip() for a in command.split("|")]
                for alias in aliases:
                    if alias.startswith(last_user_command) or (
                        isinstance(commands[command], dict)
                        and last_user_command.strip()
                        and alias.startswith(last_user_command.split()[0])
                    ):
                        shown_alias = alias
                        break
                if shown_alias is not None:
                    if not last_user_command.startswith(shown_alias + " "):
                        aliases_display = ""
                        if len(aliases) > 1:
                            aliases_display = (
                                "["
                                + "|".join(
                                    [alias for alias in aliases if alias != shown_alias]
                                )
                                + "]"
                            )
                        yield Completion(
                            shown_alias,
                            start_position=-len(last_user_command),
                            display=shown_alias + " " + aliases_display,
                            display_meta=(
                                commands[command]
                                if isinstance(commands[command], str)
                                else commands[command]["meta"]
                            ),
                        )
                    elif (
                        isinstance(commands[command], dict)
                        and commands[command]["commands"]
                    ):
                        yield from self.complete(
                            " ".join(last_user_command.strip().split()[1:]),
                            commands[command]["commands"],
                        )

    def get_completions(self, document, _):
        text = document.text
        yield from self.complete(text, self.commands, "~")


def user_prompt(default=""):
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    def prompt_continuation(width, line_number, is_soft_wrap):
        return "." * max(width - 1, 0) + " "

    return prompt_session.prompt(
        ">>> ",
        key_bindings=kb,
        multiline=True,
        wrap_lines=True,
        default=default,
        prompt_continuation=prompt_continuation,
        completer=CommandCompleter(characters=brain.gen.characters),
        complete_while_typing=True,
    )


def loading_progress(disable=True, style="white", description=""):
    bar = BarColumn(bar_width=None, pulse_style=style)
    progress = Progress(
        bar,
        disable=disable,
        transient=True,
    )

    if description:
        progress = Progress(
            bar,
            TextColumn("{task.description}", style=style),
            bar,
            disable=disable,
            transient=True,
        )

    progress.add_task(description=description, total=None)
    progress.start()
    return progress


def character_list():
    if len(brain.gen.characters.keys()) > 0:
        panel_content = Columns(
            [character for character in brain.gen.characters.keys()], expand=True
        )
    else:
        panel_content = "No characters have been shown yet."
    return Panel(panel_content, title="Characters")


def character_card(name, character, img_dir=None):
    return Panel(
        Columns(
            [
                f"[yellow link={img_dir}\\{name.replace(' ', '+')}.png]show[/]",
                brain.gen.pixelize(character["img"], ratio=14),
                Markdown("\n## Description\n" + character["prompt"]),
            ],
            equal=True,
            expand=True,
        ),
        title=name.title(),
    )


def generate_scene(img_dir):
    progress = Progress(
        TextColumn("{task.description}", style="yellow"),
        SpinnerColumn("dots", style="yellow"),
        BarColumn(bar_width=None, style="yellow"),
        MofNCompleteColumn(),
        disable=True,
        transient=True,
    )

    task = progress.add_task(description="Generating image prompt", total=2, start=True)

    with Live(progress, console=console, refresh_per_second=30, transient=True) as live:
        img = brain.generate_scene(
            created_char_hook=lambda character: progress.update(
                task,
                description=f"Generating [italic]{character}[/] prompt",
                completed=1,
                total=4,
            ),
            generating_char_image_hook=lambda character: progress.update(
                task,
                description=f"Generating [italic]{character}[/] image",
                completed=2,
                total=4,
            ),
            generated_char_image_hook=lambda character, character_img: character_img.save(
                img_dir + f'\\{character.replace(" ", "+")}.png'
            ),
            generating_scene_hook=lambda: progress.update(
                task,
                description="Generating scene",
                completed=3,
                total=4,
            ),
        )

    return brain.gen.pixelize_save_show(img, img_dir=img_dir)


def main_loop(console, brain, auto_hijack=False, auto_show=False, img_dir=None):
    user_input = ""
    while True:
        user_input = user_prompt()
        if (
            user_input.strip().startswith("~bye")
            or user_input.strip().startswith("~quit")
            or user_input.strip().startswith("~q")
            or user_input.strip().startswith("~exit")
        ):
            break

        params = [p.lower().strip() for p in user_input.strip().split("~")]
        user_input = params[0]
        ai_input = ""
        params = {
            p[0].strip(): " ".join(p[1:]).strip()
            for p in [p.split() for p in params[1:]]
        }

        auto_hijack_param = params.get("auto-hijack", params.get("ah"))
        if auto_hijack_param is not None:
            auto_hijack = not auto_hijack
            console.print(
                f"[orange]Auto-Hijack[/] {'[bold green]ON[/]' if auto_hijack else '[bold red]OFF[/]'}"
            )
            continue

        auto_show_param = params.get("auto-show", params.get("as"))
        if auto_show_param is not None:
            auto_show = not auto_show
            console.print(
                f"[orange]Show Mode[/] {'[bold green]ON[/]' if auto_show else '[bold red]OFF[/]'}"
            )
            continue

        one_time_show_param = params.get("show", params.get("s")) is not None
        if not user_input.strip() and one_time_show_param and not auto_show:
            console.print(*generate_scene(img_dir=img_dir))
            continue

        show_messages_param = params.get("messages", params.get("m"))
        if show_messages_param is not None:
            console.print(brain.ephemeral_chat_history.messages)
            continue

        characters_param = params.get("characters", params.get("chars"))
        if characters_param is not None and not user_input.strip(user_input):
            if characters_param:
                name = characters_param.strip()
                chars = {
                    char.lower(): brain.gen.characters[char]
                    for char in brain.gen.characters
                }
                if name in chars:
                    console.print(character_card(name, chars[name], img_dir=img_dir))
                else:
                    console.print(
                        f"[red]Character [bold italic]{name}[/] wasn't shown yet.[/]"
                    )
            else:
                console.print(character_list())
            continue

        hijack = params.get("hijack", params.get("h"))
        if hijack is not None:
            ai_input = hijack if hijack else "Sure, "
            user_input = user_input.replace("/hijack", "").lstrip()

        with Live(loading_progress(), console=console, refresh_per_second=30) as live:
            newline = False
            image_executor = None
            image_content = []
            for chunk, content in brain.stream(
                input=user_input,
                ai=ai_input,
                formatter=Brain.RP_FORMATTER if auto_show else None,
            ):
                if not newline:
                    console.print("")
                    newline = True

                if auto_show:
                    update = Markdown(chunk.get("response", "..."))
                    if "prompt" in chunk:
                        update = Group(
                            Markdown("> " + chunk["prompt"].replace("\n", "> ")),
                            loading_progress(
                                description="Generating Image", style="yellow"
                            ),
                            update,
                        )
                else:
                    update = Markdown(content)
                live.update(update)

                if auto_show and image_executor is None and "prompt" in content:
                    image_executor = Thread(
                        target=lambda: (
                            image_content.append(
                                brain.gen.pixelize_save_show(
                                    brain.gen.generate(content["prompt"]),
                                    img_dir=img_dir,
                                )
                            )
                        ),
                    )
                    image_executor.start()

            if image_executor is not None:
                image_executor.join()
                live.update(
                    Group(
                        *image_content[-1],
                        Rule(style="yellow"),
                        Markdown(content["response"]),
                    )
                )
                image_content.clear()

            if auto_hijack:
                content = brain.uncensor(
                    content,
                    pre_uncensor_hook=lambda: live.update(
                        loading_progress(style="red", description="UNCENSORING")
                    ),
                    uncensor_hook=lambda content: live.update(Markdown(content)),
                )

        if one_time_show_param:
            console.print(*generate_scene(img_dir=img_dir))

        console.print("")

        read_param = params.get("read", params.get("r"))
        if read_param is not None:
            brain.read_thread(content)

        if auto_show:
            brain.ephemeral_chat_history.messages[-2].content = user_input

        brain.ephemeral_chat_history.messages[-1].content = str(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="otui", description="Ollama Terminal User Interface"
    )

    parser.add_argument(
        "--auto_hijack",
        "--ah",
        action="store_true",
        default=False,
        help="Initializes otui in auto-hijack mode.",
    )

    parser.add_argument(
        "--auto_show",
        "--as",
        action="store_true",
        default=False,
        help="Initializes otui in auto-show mode.",
    )

    parser.add_argument(
        "--fast_images",
        "--fi",
        "--lcm",
        action="store_true",
        default=False,
        help="Initializes otui in fast image generation mode.",
    )

    # argument for ollama model
    parser.add_argument(
        "--model",
        "--m",
        action="store",
        default="llama3",
        help="The model to use for OTUI. Defaults to llama3.",
    )

    parser.add_argument(
        "--image_model",
        "--im",
        action="store",
        default=None,
        help="""The model to use for OTUI image generation. Defaults to None to use the same general model.
This can be helpful if your general model takes a while to load, you can put a smaller model here only for the image generation to be faster.""",
    )

    args = parser.parse_args()

    console = Console()
    prompt_session = PromptSession()
    llm = ChatOllama(model=args.model, system="", template="")
    image_llm = (
        ChatOllama(model=args.image_model, system="", template="")
        if args.image_model is not None
        else None
    )
    brain = Brain(llm, image_llm=image_llm, lcm=args.fast_images)

    with tempfile.TemporaryDirectory() as tempdir:
        main_loop(
            console,
            brain,
            auto_hijack=args.auto_hijack,
            auto_show=args.auto_show,
            img_dir=tempdir,
        )

    brain.close()
