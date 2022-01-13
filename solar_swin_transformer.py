import timm
import torch.nn as nn
import numpy as np
import timm
import torch


class SolarSwinTransformer(nn.Module):  
    def __init__(self, cfg):
        super(SolarSwinTransformer, self).__init__()
        self.hidden_layer_size = 166
        self.dropout = 0.6990787087509548
        self.pretrained_model = timm.create_model('swin_base_patch4_window7_224',
                                                  pretrained=True,
                                                  num_classes=self.hidden_layer_size)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer_size + 1, self.hidden_layer_size),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer_size,1),
            nn.LeakyReLU()
        )
        
        
    def forward(self, batch):
        """
        Batch should be in format:
        {
            'images': torch.FloatTensor((batch_size, 1, 224, 224))
            'carrington_rotation_speeds': torch.FloatTensor((batch_size, 1))
        }
        
        carrington_rotation_speed refers to the scaled olar wind speed one carrington rotation ago.
        """
        images = batch['images']
        rotation_speeds = batch['carrington_rotation_speeds']
        batch_size = images.size()[0]
        
        # Pretrained swin transformer accepts three channel images
        three_channel = torch.stack(3 * [images], dim=2).squeeze(1)
        
        # Model learns optimal initial normalisation
        normalized_images = self.bn(three_channel)
        
        # Get image features
        image_features = self.pretrained_model.forward(normalized_images).view(batch_size, -1)
        
        #Concat with the carrington rotation
        features_with_speeds = torch.cat((image_features, rotation_speeds), dim=1)
        
        output = self.fc(features_with_speeds)
        return output
    