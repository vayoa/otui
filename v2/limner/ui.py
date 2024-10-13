import gradio as gr
from eyes import Eyes
from limner import main_runner as mr, drawn as d

DEFAULT_SETTINGS = r"""
{
  "canvas_width": 1080,
  "canvas_height": 1920,

  "panel_min_width_percent": 0.21,
  "panel_min_height_percent": 0.117,
  
  "image_zoom": "1",
  "border_width": 5,
  "default_bubble": {
    "font_size": 16,
    "bubble_color": "(20, 20, 20)",
    "text_color": "white",
    "font": "C:\\Windows\\Fonts\\Arial.ttf"
  },

  "prompt_prefix": "score_9, score_8_up, score_7_up, speech bubbles",
  "negative_prompt": "score_6, score_5, score_4",

  "embeds": {
    "concepts": {
    "spysuit": "black spy suit, tech"
    },
    "characters": {
      "test": {
        "tags": "guy, male, short black hair, white pants, white shirt",
        "bubble": {
          "bubble_color": "(200, 200, 200)",
          "text_color": "black"
        }
      }
    }
  }
}
"""
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

[test] weaving his hands
"Heyy"

split: h

[test] lifting an apple
"yo wtf"-Page1

location: pov doorway

[test] weaving his hands
"Heyy"

split: h

[test] lifting an apple
"yo wtf"
"""


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        last_script = gr.State({})
        with gr.Row():
            with gr.Tab("Script"):
                script = gr.TextArea(
                    lines=20,
                    label="start with --api for this extension to work.",
                    value=DEFAULT_SCRIPT,
                    placeholder=DEFAULT_SCRIPT_HINT,
                )

            with gr.Tab("Settings"):
                settings = gr.Code(
                    language="json",
                    lines=20,
                    show_label=False,
                    value=DEFAULT_SETTINGS,
                )

            with gr.Column():
                with gr.Row():
                    btn = gr.Button("Generate", variant="primary")
                    page = gr.Number(label="Page", value=1, minimum=1)
                    panel = gr.Number(label="Panel", value=1, minimum=1)
                    chk = gr.Checkbox(label="Full Page", value=True)

                gallery = gr.Gallery(
                    label="Dummy Image",
                    show_label=False,
                    object_fit="cover",
                )

        btn.click(
            generate,
            inputs=[script, last_script, settings, chk, page, panel],
            outputs=[gallery, last_script],
        )

        return [(ui_component, "Castadi", "castadi_tab")]


def remove_json_comments(json):
    return "\n".join([l for l in json.split("\n") if not l.strip().startswith("//")])


def generate(script, last_script, settings, chk, page, panel):
    if script and settings:
        if chk:
            last_script, location = None, None
        else:
            last_script, location = d.script_from_dict(last_script), (page, panel)

        settings = remove_json_comments(settings)
        _, results, new_script = mr.generate(
            script,
            settings,
            last_script=last_script,
            location=location,
        )
        return [results, d.script_to_dict(new_script)]
    else:
        return [[], {}]


EYES = Eyes(default_checkpoint="waiANINSFWPONYXL_v70.safetensors")

if __name__ == "__main__":
    app = on_ui_tabs()
    app[0][0].launch()
    EYES.close()
