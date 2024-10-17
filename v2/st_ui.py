import streamlit as st
from streamlit_components.components import settings_sidebar
from code_editor import code_editor
from limner import main_runner as mr, drawn as d

# from limner import main_runner as mr, drawn as d
DEFAULT_SCRIPT_HINT = """\
Your script, example:
-Page1

location: public park, yard, outside, park

[misty] weaving, saying hey, happy, smiling
"Heyy!!"

"This is an example"
"of what castadi can do!"

[misty] curious, surpries, happy, question
"What do you think?"
"""
DEFAULT_SCRIPT = """
-Page1

location: pov doorway

[rin] weaving her hands
"Heyy"

split: h

[rin] lifting an apple
"yo wtf"
"""


def generate(script, last_script, settings, chk, page, panel):
    if script and settings:
        if chk:
            last_script, location = None, None
        else:
            last_script, location = d.script_from_dict(last_script), (page, panel)

        _, results, new_script = mr.generate(
            script,
            settings,
            last_script=last_script,
            location=location,
        )
        return [results, d.script_to_dict(new_script)]
    else:
        return [[], {}]


def wide():
    st.set_page_config(layout="wide")


wide()

last_script = st.session_state.get("last_script", {})
settings = settings_sidebar()

left, right = st.columns(2)
with left:
    with st.form("script"):
        row = st.columns([0.2, 0.1, 0.2, 0.1, 0.4], vertical_alignment="center")
        page = row[0].number_input(
            "Page",
            min_value=1,
            label_visibility="collapsed",
        )
        row[1].write("Page")
        panel = row[2].number_input(
            "Panel",
            min_value=0,
            label_visibility="collapsed",
        )
        row[3].write("Panel")
        full_chk = panel == 0
        submitted = row[4].form_submit_button(
            f"Generate {'Page' if full_chk else ''}",
            type="primary",
            use_container_width=True,
        )

        script = code_editor(
            code=DEFAULT_SCRIPT,
            lang="python",
            height=[20, 20],  # type: ignore
            options={"wrap": True},
            replace_completer=True,
            buttons=[
                {
                    "name": "Copy",
                    "feather": "Copy",
                    "hasText": False,
                    "alwaysOn": True,
                    "commands": ["copyAll"],
                    "style": {"top": "0.46rem", "right": "0.4rem"},
                }
            ],
            completions=[
                {
                    "caption": completion,
                    "meta": meta,
                    "value": value,
                    "name": completion,
                    "score": 100,
                }
                for completion, meta, value in [
                    ("location", "embed", "location: "),
                    ("split", "operator", "split: "),
                    ("splith", "horizontal split", "split: h"),
                    ("splitv", "vertical split", "split: v"),
                    *[
                        (concept, "concept", concept)
                        for concept in settings["embeds"]["concepts"].keys()
                    ],
                    *[
                        (char, "character", char)
                        for char in settings["embeds"]["characters"].keys()
                    ],
                ]
            ],
            props={"enableSnippets": True, "placeholder": DEFAULT_SCRIPT_HINT},
            allow_reset=True,
            key="script-editor",
            response_mode=["debounce", "blur"],  # type: ignore
        )
        script = script["text"]

        if submitted:
            try:
                images, new_last_script = generate(
                    script, last_script, settings, full_chk, page, panel
                )
                st.session_state["last_script"] = new_last_script
                st.session_state["images"] = images
            except KeyError as key:
                st.error(f"No such embed as {key}")

with right.container(border=True):
    images = st.session_state.get("images")
    if images:
        page_selector = page if page <= len(images) else len(images)
        output = st.image(
            images[page_selector - 1],
            width=425,
        )

    # output = carousel(
    #     [
    #         dict(
    #             title="",
    #             text=str(i),
    #             img=img,
    #         )
    #         for i, img in enumerate(
    #             images  # type: ignore
    #             # [
    #             #     "./limner/images/result-1-4.png",
    #             #     "./limner/images/result-2-3.png",
    #             # ]
    #         )
    #     ],  # type: ignore
    #     width=0.64,
    #     container_height=575,
    #     wrap=False,
    #     interval=None,  # type: ignore
    # )
