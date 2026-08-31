from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn

class lstm_encoder(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Linear(input_size, 64)
        self.hidden_size = 128
        self.num_layers = 1

        # define LSTM layer
        self.lstm = nn.LSTM(64, hidden_size=128,  # input æ˜?(batch_size, sequence_size, input_size)
                            num_layers=1, batch_first=True)

        # define activation:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x_input):
        embedded = self.embedding(x_input)
        embedded = self.leaky_relu(embedded)   
        lstm_out, hidden = self.lstm(embedded) #output:(batch, SeqL, HiddenSize) (128, 5, 128); hidden:(1, Bsz:128, hidden128)

        return lstm_out, hidden

class ACNet(nn.Module):

    ## Initialization
    def __init__(self,args, enc_input_size, hidden_size=128, output_timestep = 25, output_size = 1, ):
        super(ACNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']
        

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']
        self.use_maneuvers = args['use_maneuvers']
        self.joint_classes = args['joint_classes'] 
        self.enc_input_size = enc_input_size
        self.encoder = lstm_encoder(self.enc_input_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        ## Sizes of network layers
        self.in_length = args['in_length']
        self.input_embedding_size = args['input_embedding_size']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.num_joint_classes = args['joint_classes']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.batch_size = args['batch_size']
        self.val_batch_size = args['val_batch_size']

        # Decoder LSTM
        if self.use_maneuvers:
            #self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            #self.dec_lstm = torch.nn.LSTM(self.hidden_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
            self.dec_lstm = torch.nn.LSTM(self.hidden_size + self.joint_classes, self.decoder_size, batch_first=True) # input size and output size

        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, self.output_size) # ()
        self.op_joint = torch.nn.Linear(self.hidden_size, self.num_joint_classes) # hidden size is 128, op_joint is (1, 128, 3)
        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)


    ## Forward Pass
    def forward(self,hist, joint_enc): #joint_enc is (bsz,3)
        input_batch = hist  
        enc_out, enc = self.encoder(input_batch)  # enc_out is (128,5,128); enc[0] is (1, bsz=128, hidden=128)
        #print(enc.shape)
        #enc = torch.squeeze(enc[0])
        #enc=enc.view()
        #encodeing shape: (32,112)

        if self.use_maneuvers:
            ## Maneuver recognition:
            joint_pred = self.softmax(self.op_joint(enc[0])) #(1,128,3)

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                joint_enc = joint_enc.view(1,self.batch_size,3)
                #enc_temp = enc[0]
                dec_input = torch.cat((enc[0], joint_enc), dim=2)  # (bsz,131)
                fut_pred = self.decode(dec_input) # 
                return fut_pred, joint_pred
            else:
                joint_enc = joint_enc.view(1,self.val_batch_size,3)
                dec_input = torch.cat((enc[0], joint_enc), 2)  # (1,128,131)
                fut_pred = self.decode(dec_input) # 
                return fut_pred, joint_pred                
                # fut_pred = []
                # ## Predict trajectory distributions for each maneuver class
                # for k in range(self.num_joint_classes):

                #     joint_enc_tmp = torch.zeros_like(joint_enc)
                #     joint_enc_tmp[:, k] = 1
                #     enc_tmp = torch.cat((enc_out, joint_enc_tmp), 1)
                #     fut_pred.append(self.decode(enc_tmp))
                # return fut_pred, joint_pred
        else:
            fut_pred = self.decode(enc_out) #(25,32,5)
            return fut_pred


    def decode(self,enc): # enc is (1,128,131)
        enc_ = enc.repeat(self.out_length, 1, 1) # (5,128,131)
        enc_ = enc_.permute(1, 0, 2) #(5,128,131)
        h_dec, _ = self.dec_lstm(enc_) # h_dec is 128, 5, decaoder_size:256
        #h_dec = h_dec.permute(1, 0, 2)  #(32,25,128)
        fut_pred = self.op(h_dec)  #128, 5, outputSize:1 
        #fut_pred = fut_pred.permute(1, 0, 2) #(25,32,5)
        return fut_pred # (128,5,1)