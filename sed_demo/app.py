import gradio as gr

from sed_demo.sound_event_detection import sed

title = "EnviroX Sound Event Detection"
description = "Sound classification demo based on PANNs (pretrained audio neural networks). Record a sound from your device to analyse it (remember to press stop recording when done!)."
article = "The output shows the probability (0-1) of the sounds it has detected during the recording."

recording_demo = gr.Interface(
    title=title,
    description=description,
    article=article,
    allow_flagging="never",
    fn=sed,
    inputs=[gr.Audio(label="Recording", source="microphone", type="filepath")],
    outputs=[
        gr.TimeSeries(label="Sound content"),
        gr.Text(label="Status Message", value="Please record a sound to proceed."),
    ],
    live=True,
    analytics_enabled=True,
)

def start():
    recording_demo.queue()
    recording_demo.launch(server_name="0.0.0.0")
