from transformers import T5Model, BartModel, T5ForConditionalGeneration
import torch
from torch import nn
import torch.nn.functional as F

class T5PromptTuning(T5ForConditionalGeneration):

    def __init__(self) -> None:
        super(T5PromptTuning, self).__init__()