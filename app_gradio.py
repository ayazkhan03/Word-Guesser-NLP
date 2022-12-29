import gradio as gr
from core.inference import inference
from core.tokenizer import get_num_tokens_padding_idx
from train import Experiment
from core.model import Nueral_Net
import torch

num_tokens, padding_idx = get_num_tokens_padding_idx("./data/tokenizer.pth")
gru = Nueral_Net(num_tokens, padding_idx)
tokenizer = torch.load("./data/tokenizer.pth")

model = Experiment(gru, padding_idx, num_tokens)
model.load_state_dict(torch.load("./logs/version_3/checkpoints/epoch=500-step=60035.ckpt", map_location="cpu")["state_dict"])

def spell_correction(input_masked_word, input_description):
    cleaned = inference(input_masked_word, input_description, tokenizer, model)
    return cleaned

app = gr.Interface(
    title = "Saama Word Guesser ❓",
    fn=spell_correction,
    inputs=["text", "text"],
    outputs="text"
)


app.launch()