import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normal form 1
class FrequenceNet(nn.Module):
  def __init__(self, extract_model, restore_model, input_dim, compress_ratio):
    super().__init__()
    self.extract_model = extract_model
    self.restore_model = restore_model
    assert not input_dim%compress_ratio, ValueError('compress_ratio must be divisible by input_dim.')
    self.compress_dim = int(input_dim//compress_ratio)
    self.full_connect_encoder = nn.Linear(input_dim, self.compress_dim)
    self.full_connect_decoder = nn.Linear(self.compress_dim, input_dim)
    
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)     

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
  

# Normal form 2
class ExtractionStrategies:
    def __init__(self, input_dim, compress_ratio) -> None:
        self.input_dim = input_dim
        self.compress_ratio = compress_ratio
        self.compress_dim = int(input_dim//compress_ratio)
        
    def equidistantExtraction(self):
        return lambda x: x[:, : :self.compress_ratio, :].clone()
    

class ChoseNet(nn.Module):
    def __init__(self, strategy_f, restore_model, input_num, restore_num):
        super().__init__()
        self.strategy_f = strategy_f
        self.restore_model = restore_model
        self.full_connect_decoder = nn.Linear(input_num, restore_num)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)     
        
    def forward(self, data):
        data = self.strategy_f(data)
        restore_feature_raw = self.full_connect_decoder(data.transpose(-1, -2)).transpose(-1, -2)
        restore_feature = self.restore_model(restore_feature_raw)
        return restore_feature