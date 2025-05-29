import torch
import torchaudio
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
#from speechbrain.pretrained import EncoderClassifier


def PrepareArchitecture(model, trainable_parameter_list):
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if any(layer in name for layer in trainable_parameter_list):
            param.requires_grad = True
    return model

def PrepareArchitecture1(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def PrepareArchitecture2(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

class PreTrainedModels(nn.Module):
    def __init__(self, name="microsoft/wavlm-large", *args, **kwargs):        
        super().__init__(*args, **kwargs)        
        self.name = name
        try:
            self.model = AutoModel.from_pretrained(name)  # Load model using Hugging Face
            self.hidden_size = self.model.config.hidden_size  # Extract hidden size dynamically
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            self.model = None
            self.hidden_size = None  # Handle error case

    def forward(self, INPUT, attention_mask=None):
        if self.model is None:
            raise RuntimeError("Pretrained model failed to load.")

        output = self.model(INPUT, attention_mask=attention_mask)
        

        # Ensure compatibility with different model output types
        if isinstance(output, tuple):  # Some models return tuples
            return output[0]  # First element is usually last_hidden_state
        elif hasattr(output, "last_hidden_state"):  # Transformer models
            return output.last_hidden_state
        else:
            raise RuntimeError("Unexpected output format from the model.")


class PreTrainedModelsSpeechBrain(nn.Module):
    def __init__(self, name="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", *args, **kwargs):        
        super().__init__(*args, **kwargs)
        self.name = name
        try:
            # Load model using SpeechBrain's API
            self.model = EncoderClassifier.from_hparams(
                source=name,
                savedir="pretrained_models/speechbrain_emotion_recognition"
            )
            # Determine hidden size
            modules_list = list(self.model.modules())
            if len(modules_list) > 0 and hasattr(modules_list[0], "config"):
                self.hidden_size = modules_list[0].config.hidden_size
            else:
                self.hidden_size = 768  # Fallback value
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            self.model = None
            self.hidden_size = None

        # Determine which encoder module to use from self.model.mods
        mods_keys = list(self.model.mods.keys()) if self.model is not None else []
        if "encoder" in mods_keys:
            self.encoder_module = self.model.mods["encoder"]
        elif "wav2vec2" in mods_keys:
            self.encoder_module = self.model.mods["wav2vec2"]
        else:
            raise RuntimeError("No suitable encoder module found in model.mods. Available keys: " + str(mods_keys))

    def forward(self, INPUT, attention_mask=None):
        if self.model is None:
            raise RuntimeError("Pretrained model failed to load.")
        # Use the pre-determined encoder module
        output = self.encoder_module(INPUT)
        # Depending on your downstream needs, you might process 'output' (e.g., squeeze dimensions)
        return output

class PreTrainedModelsEmotion2Vec(nn.Module):
    def __init__(self, name="iic/emotion2vec_plus_base", *args, **kwargs):
        """
        Initialize the Emotion2Vec model.
        Replace "your_org/emotion2vec-model" with the correct Hugging Face model identifier.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        try:
            # Load using the standard Hugging Face API.
            self.model = AutoModel.from_pretrained(name)
            # Retrieve the hidden size from the model configuration.
            self.hidden_size = self.model.config.hidden_size
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            self.model = None
            self.hidden_size = None

    def forward(self, INPUT, attention_mask=None):
        if self.model is None:
            raise RuntimeError("Pretrained model failed to load.")
        # Forward pass through the Emotion2Vec model.
        output = self.model(INPUT, attention_mask=attention_mask)
        # If the output is a tuple, assume the first element is the last hidden state.
        if isinstance(output, tuple):
            output = output[0]
        return output

# from transformers import AutoModel
# import torch.nn as nn

# class PreTrainedModels(nn.Module):
#     def __init__(self, name="microsoft/wavlm-large", *args, **kwargs):        
#         super().__init__(*args, **kwargs)        
#         self.name = name
#         try:
#             self.model = AutoModel.from_pretrained(name)  # Load model using Hugging Face
#             self.hidden_size = self.model.config.hidden_size  # Extract hidden size dynamically
#         except Exception as e:
#             print(f"Error loading model {name}: {e}")
#             self.model = None
#             self.hidden_size = None  # Handle error case

#     def forward(self, INPUT, attention_mask=None, output_hidden_states=False):
#         if self.model is None:
#             raise RuntimeError("Pretrained model failed to load.")

#         # Forward pass with the correct `output_hidden_states` flag
#         output = self.model(INPUT, attention_mask=attention_mask, output_hidden_states=output_hidden_states)

#         # Ensure compatibility with different Hugging Face model output formats
#         if output_hidden_states:
#             if hasattr(output, "hidden_states"):
#                 return output.hidden_states  # ✅ Returns all hidden states from all layers
#             else:
#                 raise RuntimeError("Model does not support `output_hidden_states=True`.")

#         if isinstance(output, tuple):  # Some models return tuples
#             return output[0]  # ✅ First element is usually `last_hidden_state`
#         elif hasattr(output, "last_hidden_state"):  # Transformer-based models
#             return output.last_hidden_state
#         else:
#             raise RuntimeError("Unexpected output format from the model.")




class DeepAudioNetEmotionClassification(nn.Module):
    def __init__(self, input_dim, conv_channels=[64, 128], fc_hidden=256, output_dim=1, dropout=0.5):
        """
        Args:
            input_dim (int): Dimensionality of the input feature vector (e.g., 2*feat_dim).
            conv_channels (list): List with the number of channels for each conv block.
            fc_hidden (int): Hidden dimension for the fully connected layer.
            output_dim (int): Output dimension (e.g., 1 for regression or number of classes for classification).
            dropout (float): Dropout probability.
        """
        super(DeepAudioNetEmotionClassification, self).__init__()
        self.input_dim = input_dim
        
        # First convolutional block: input is treated as [batch, 1, input_dim]
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_channels[0],
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # reduces length by factor 2
            nn.Dropout(dropout)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1],
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # reduces length by another factor of 2
            nn.Dropout(dropout)
        )
        
        # After two pooling layers, the sequence length becomes input_dim//4
        pooled_dim = input_dim // 4
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[1] * pooled_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, output_dim)
        )
    
    def forward(self, x):
        # x is expected to be [batch, input_dim]
        # Unsqueeze to create a channel dimension: [batch, 1, input_dim]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten for the FC layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EmotionClassification(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EmotionClassification, self).__init__()
        input_dim = args[0]
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        p = kwargs.get("dropout", 0.5)

        self.fc=nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
            )
        ])
        for lidx in range(num_layers-1):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
                )
            )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.inp_drop = nn.Dropout(p)
    def get_repr(self, x):
        h = self.inp_drop(x)
        for lidx, fc in enumerate(self.fc):
            h=fc(h)
        return h
    
    def forward(self, x):
        h=self.get_repr(x)
        result = self.out(h)
        return result
    
class Basic_Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.l1 = nn.Linear(12,1)
    
    def forward(self, INPUT):
        return self.l1(INPUT)
    
class SER_Base_model(nn.Module):
    def __init__(self, in_channel=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.embedding_1 = nn.Sequential(
            nn.Conv1d(in_channel, 512, 3),
            nn.Conv1d(512, 128, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64,3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64,1,3)
        )
        
        self.embedding_2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(128),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64,8)
        )
    
    def forward(self, INPUT):
        # out size must be (8,8), input size will be (bs, seq, channel)
        INPUT = INPUT.permute(0,2,1)
        o = self.embedding_1(INPUT)
        o1 = self.embedding_2(o)
        return o1.squeeze(1)
    
    
    
class FinalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=8, dropout=0.5):

        super(FinalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    
class EmotionRegression1(nn.Module):
    def __init__(self, dh_input_dim, head_dim,  output_dim, num_layers,dropout=0.5):
        super(EmotionRegression1, self).__init__()
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dh_input_dim, head_dim),
                nn.LayerNorm(head_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        for _ in range(num_layers - 1):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(head_dim, head_dim),
                    nn.LayerNorm(head_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.out = nn.Sequential(
            nn.Linear(head_dim, output_dim)
        )
        self.inp_drop = nn.Dropout(dropout)

    def get_repr(self, x):
        h = self.inp_drop(x)
        for layer in self.fc:
            h = layer(h)
        return h

    def forward(self, x):
        h = self.get_repr(x)
        result = self.out(h)
        return result
    
    
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        weights = self.softmax(self.fc1(x))  # Compute expert weights
        return weights


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + identity

class ImprovedFinalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=8, dropout=0.5, num_residual_blocks=2):
        super(ImprovedFinalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Create a sequence of residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_residual_blocks)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.res_blocks(out)
        out = self.fc_out(out)
        return out

class ImprovedGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, dropout=0.5):
        super(ImprovedGatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # The final layer outputs a logit for each expert which is converted into a probability distribution.
        weights = torch.softmax(self.fc2(x), dim=1)
        return weights
