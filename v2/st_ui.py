import streamlit as st
from streamlit_ace import st_ace
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
settings = {}

with st.sidebar:
    st.title("Settings")
    defaults_tab, embeds_tab = st.tabs(("Defaults", "Embeds"))

    with defaults_tab:
        st.subheader("Canvas")
        left, right = st.columns(2)
        settings["canvas_width"] = left.number_input(
            "Width", value=1280, min_value=512, step=128
        )
        settings["canvas_height"] = right.number_input(
            "Height", value=1920, min_value=512, step=128
        )

        st.subheader("Panel")
        left, right = st.columns(2)
        settings["image_zoom"] = left.number_input(
            "Image Zoom",
            value=1.0,
            min_value=0.2,
            max_value=1.0,
            step=0.2,
        )
        settings["border_width"] = right.slider(
            "Border Width",
            value=5,
            min_value=0,
            max_value=15,
        )

        left, right = st.columns(2)
        settings["panel_min_width_percent"] = left.slider(
            "Min Width",
            value=0.3,
            format="%0.3f",
            min_value=0.0,
            max_value=1.0,
        )
        settings["panel_min_height_percent"] = right.slider(
            "Min Height",
            value=0.3,
            format="%0.3f",
            min_value=0.0,
            max_value=1.0,
        )

        st.divider()
        st.header("Default Bubble")
        left, middle, right = st.columns(3)
        default_bubble = {}
        default_bubble["font_size"] = left.number_input(
            "Font Size",
            value=16,
            min_value=4,
            max_value=40,
        )
        default_bubble["bubble_color"] = middle.color_picker("Bubble", value="#141414")
        default_bubble["text_color"] = right.color_picker("Text", value="#141414")

        default_bubble["font"] = "C:\\Windows\\Fonts\\Arial.ttf"
        settings["default_bubble"] = default_bubble

    with embeds_tab:
        embeds = {}

        settings["prompt_prefix"] = st.text_input(
            "Positive Prefix",
            value="score_9, score_8_up, score_7_up, speech bubbles",
        )
        settings["negative_prompt"] = st.text_input(
            "Negative Prefix", value="score_6, score_5, score_4"
        )

        with st.expander("Concepts"):
            concepts = st.data_editor(
                dict(Name=["Concept1"], Tags=[""]),
                column_config={
                    "Name": st.column_config.TextColumn(required=True, width="small"),
                    "Tags": st.column_config.TextColumn(required=True, width="large"),
                },
                use_container_width=True,
                num_rows="dynamic",
                key="concepts-dataeditor",
            )
            embeds["concepts"] = {
                name: tags for name, tags in zip(concepts["Name"], concepts["Tags"])
            }

        with st.expander("Characters"):
            characters = st.data_editor(
                dict(Name=["rin"], Tags=["tohsaka rin"]),
                column_config={
                    "Name": st.column_config.TextColumn(required=True, width="small"),
                    "Tags": st.column_config.TextColumn(required=True, width="large"),
                },
                use_container_width=True,
                num_rows="dynamic",
                key="characters-dataeditor",
            )
            embeds["characters"] = {
                name: {"tags": tags}
                for name, tags in zip(characters["Name"], characters["Tags"])
            }

        settings["embeds"] = embeds

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

        script = st_ace(
            value=DEFAULT_SCRIPT,
            height=492,
            placeholder=DEFAULT_SCRIPT_HINT,
            language="actionscript",
            wrap=True,
            theme="twilight",
            min_lines=0,
            auto_update=True,
        )

        if submitted:
            try:
                import random

                print("hey", random.random())
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
