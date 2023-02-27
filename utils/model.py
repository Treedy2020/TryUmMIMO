import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FrequenceNet(nn.Module):
  def __init__(self, extract_model, restore_model, input_dim, compress_ratio, isTransformer=False):
    super().__init__()
    self.extract_model = extract_model
    self.restore_model = restore_model
    assert not input_dim%compress_ratio, ValueError('compress_ratio must be divisible by input_dim.')
    self.compress_dim = int(input_dim//compress_ratio)
    self.full_connect_encoder = nn.Linear(input_dim, self.compress_dim)
    self.full_connect_decoder = nn.Linear(self.compress_dim, input_dim)
    self.isTransformer=isTransformer

  def forward(self, data):
    extract_feature = self.extract_model(data)
    compress_feature = self.full_connect_encoder(extract_feature)
    restore_feature_raw = self.full_connect_decoder(compress_feature)
    restore_feature = self.restore_model(restore_feature_raw)
    return restore_feature
  

class FeatureExtractModel:
  def __init__(self, input_dim):
    super().__init__()
    self.input_dim = input_dim
    
  def getLinear(self):
    return nn.Linear(self.input_dim, self.input_dim)
  
  def getTransformer(self):
    return nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=1, dropout=0., batch_first=True)