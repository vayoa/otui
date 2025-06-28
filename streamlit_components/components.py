import streamlit as st


def settings_sidebar():
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
            default_bubble["bubble_color"] = middle.color_picker(
                "Bubble", value="#141414"
            )
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
                    dict(Name=[""], Tags=[""]),
                    column_config={
                        "Name": st.column_config.TextColumn(
                            required=True, width="small"
                        ),
                        "Tags": st.column_config.TextColumn(
                            required=True, width="large"
                        ),
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
                        "Name": st.column_config.TextColumn(
                            required=True, width="small"
                        ),
                        "Tags": st.column_config.TextColumn(
                            required=True, width="large"
                        ),
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
    return settings
