import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from huggingface_hub import hf_hub_downloader

@st.cache_resource
def load_model():
    path = hf_hub_downloader('', 'weights_model_convnext_small_v2.pt')

    model = models.convnext_small(pretrained=False)
    model.classifier[2] = nn.Linear(in_features=768, out_features=4, bias=True)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()

    return model

model = load_model()