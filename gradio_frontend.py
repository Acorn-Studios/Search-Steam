import gradio as gr
from use import *

global ST
ST = SteamTransformer()

def greet(prompt, n=1):
    global ST
    r=ST.predict(prompt)
    output = []
    for i in range(n):
        output.append([
            r["appid"],
            r["url"],
        ])
    return output

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
            n = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of results")
            submit_btn = gr.Button(value="Calculate")
        with gr.Column():
            result = gr.List()

    submit_btn.click(
        greet, inputs=[prompt,n], outputs=[result], api_name=False
    )

if __name__ == "__main__":
    demo.launch(show_api=False)