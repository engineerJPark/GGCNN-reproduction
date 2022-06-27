import torch
import torch.nn as nn

class GGCNN(nn.Module):
    def __init__(self, input_channel=1, filter_sizes=[32,16,8,8,16,32], kernel_sizes=[9, 5, 3, 3, 5, 9], strides=[3, 2, 2, 2, 2, 3], paddings=[3, 2, 1, 1, 2, 3]):
        super().__init__()
        # input channel은 depth image라서 1을 기본으로 지정한다.
        # input image H W = 300 300

        self.conv1 = nn.Conv2d(input_channel, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=paddings[0]) # output 100 100
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=paddings[1]) # output 50 50
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=paddings[2]) # output 25 25
        self.conv_trans1 = nn.ConvTranspose2D(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=paddings[3], output_padding=1) # output 50 50
        self.conv_trans2 = nn.ConvTranspose2D(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=paddings[4], output_padding=1) # output 100 100
        self.conv_trans3 = nn.ConvTranspose2D(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=paddings[5], output_padding=1) # output 301 301. 의도적으로 1 크게 함
        self.relu = nn.ReLU()

        self.body = nn.Sequential( # input & output 300 300
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.conv_trans1, self.relu,
            self.conv_trans2, self.relu,
            self.conv_trans3, self.relu
        ) 

        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2) # output 300 300
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2) # output 300 300
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2) # output 300 300
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2) # output 300 300

    def forward(self, x):
        x = self.body(x)
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
    
    def loss(self, x_data, y_data):
        pos_real, cos_real, sin_real, width_real = y_data
        pos_pred, cos_pred, sin_pred, width_pred = self.forward(x_data)
        
        # use L2 loss
        loss = nn.MSELoss()
        pos_loss, cos_loss, sin_loss, width_loss = loss(pos_pred, pos_real), loss(cos_pred, cos_real), loss(sin_pred, sin_real), loss(width_pred, width_real)

        return {
            'total_loss': pos_loss + cos_loss + sin_loss + width_loss,
            'loss': {
                'pos_loss': pos_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
