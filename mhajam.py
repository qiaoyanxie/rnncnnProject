import torch
import torch.nn as nn
import numpy as np
from torchvision.models.resnet import resnet50

class MHAJAM(nn.Module):
    def __init__(self, modelParams):
        super(MHAJAM, self).__init__()
        self.modelParams = modelParams
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lstmEnc = nn.LSTM(modelParams["input_embedding_size"], modelParams["encoder_size"], 1, batch_first=True)

        # CNN for the raster
        resnet50Pretrained = resnet50(pretrained=True)

        num_history_channels = modelParams["history_num_frames"]
        num_in_channels = 5 + 2 * num_history_channels
        resnet50Pretrained.conv1 = nn.Conv2d(
            num_in_channels,
            resnet50Pretrained.conv1.out_channels,
            kernel_size=modelParams["rasterCNN_conv1_kernel_size"],
            stride=modelParams["rasterCNN_conv1_stride"],
            padding=modelParams["rasterCNN_conv1_padding"],
            bias=False,
        )

        self.rasterCNN = torch.nn.Sequential(*(list(resnet50Pretrained.children())[:-4]))
        # #MHA
        self.attn = nn.MultiheadAttention(embed_dim=64, kdim=784, vdim=784, num_heads=modelParams["num_heads"])
        self.weight_keys = nn.Conv2d(576, modelParams["cnn_out_channels"], kernel_size=1)
        self.weight_values = nn.Conv2d(576, modelParams["cnn_out_channels"], kernel_size=1)
        self.query_fc = nn.Linear(64, modelParams["q_out_channels"]) #FC layer applied to queries
        
        self.lstmDec = nn.LSTM(68, modelParams["decoder_size"], 1) #changed this to 128 bc now zl cat with other thing should be 128
        self.lstmDecFc = nn.Linear(68, 2) #x, y
        self.lstmOutputFc = nn.Linear(modelParams["decoder_size"], 68)
        
        # #Probability Generation
        self.attn_fc1 = nn.Linear(68*modelParams["num_heads"], modelParams["prob_fc1_out"]) 
        self.attn_fc2 = nn.Linear(modelParams["prob_fc1_out"], modelParams["num_heads"])

        # #S_i^t has dimension (2)
        # #e_i^t has dimension (32)
        # # rawInput has dimension (2*history*carsOnScreen)
        self.trajectoryfc = nn.Linear(modelParams["input_size"], modelParams["input_embedding_size"])

        # # Activations:
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        """
        B: Batch
        N: num agents
        H: #history frames
        agentsHist: (B, N, H, 3) tensor
        lengths: [B]
        rastImg: (B, 5+2*Hist, rastWidth, rastHeight)
        """
        agentsHist, lengths, rastImg, _, _, agentFromRaster = inputs
        B, N, H, _ = agentsHist.shape
        #raster image is just batch*224*224
        # svHistory tensor
        """
            [
                v0[
                    t0[x, y],
                    t1[x, y]
                ]
                v1[
                    t0[x, y],
                    t1[x, y]
                ]
            ]
        """
        #tvHistory (2*history)

        #reference frames... for the LSTM we want the agent reference frame.  Then we convert to raster frames

        #scene 0, frame 0, track vehicle 2
        #scene 5, frame 3, track vehicle 2
        # print(rastImg.shape)
        rasterFeatures = self.rasterCNN(rastImg) #should be (B, 512, 28, 28)
        # print(rasterFeatures.shape)


        # (B*N, H, 3)
        # (B*N, H, 32)
        # (B, N, H, 32)
        
        # for loop insert N into social tensor
        # (B, social tensor)

        # agentsHistBN = agentsHist.view(B*N, H, 3) #(B, N, H, 3) -> (B*N, H, 3)
        agentsHistBNOrigin = agentsHist.clone().detach()
        agentsHistBNOrigin = agentsHistBNOrigin.to(self.device)
        agentsHistBNOrigin = agentsHistBNOrigin.view(B*N, H, 3)
        agentsHistBNOrigin[:, :, :2] = agentsHistBNOrigin[:, :, :2] - self.modelParams['raster_img_center_offset']
        _, (positionEncoding, _) = self.lstmEnc(self.leaky_relu(self.trajectoryfc(agentsHistBNOrigin))) # (B*N, H, 32)

        #position encoding after .view is (B, N, 64)

        # print("position encoding shape: ", positionEncoding.shape)
        tvEncoding = positionEncoding.view(B, N, self.modelParams["encoder_size"])[:, 0, :]
        svEncoding = positionEncoding.view(B, N, self.modelParams["encoder_size"])[:, 1:, :] #
        

        #tvEncoding has size (encoderSize)
        #positionEncoding has size(agents*encoderSize)
        #self.leaky_relu(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2]))

        svPositionsAtT0 = agentsHist[:, 1:, 0, :]
        # socialTensor = rawToRasterScaleSocialState(rastImg.shape, rastImg.shape, svPositionsAtT0, svEncoding, lengths)
        socialTensor = rawToRasterScaleSocialState(rastImg.shape, rasterFeatures.shape, svPositionsAtT0, svEncoding, lengths - 1)

        # #assuming social tensor is (channels, M, N)
        socialTensor = torch.cat((socialTensor, rasterFeatures), dim=1)
        # #(encodeSize+rasterFeaturesChannels, M, N)

        # #keys: (num1x1Convs, M, N)
        # #values:(num 1x1 Convs, M, N)
        # #queries:()
        keys = self.weight_keys(socialTensor) #5, 576, 64, 64
        values = self.weight_values(socialTensor)
        # print("key shape", keys.shape)
        # print("value shape", values.shape)
        keys = keys.permute(1, 0, 2, 3)
        keys = keys.view((self.modelParams["cnn_out_channels"], B, 784)) #784 = 28*28
        values = values.permute(1, 0, 2, 3)
        values = values.view((self.modelParams["cnn_out_channels"], B, 784))
        # print("new key shape", keys.shape)
        # print("new value shape", values.shape)

        queries = self.query_fc(tvEncoding)
        # print("queries shape: ", queries.shape)
        queries = queries.unsqueeze(0)
        # print('new q shape', queries.shape)
        # print("great success")
        tvEncoding = tvEncoding.unsqueeze(0)
        attn_output, _ = self.attn(queries, keys, values)
        head_size = int(queries.shape[2] / self.modelParams["num_heads"]) #embed_dim / num_heads
        # #simulate l attention heads	
        decoded_outputs_t = []
        # print('tgpos shape', targetPos.shape)
        initPos = agentsHist[:,0, 0, 0:2].unsqueeze(1)
        # print('initpos shape', initPos.shape)
        # fullPos = torch.cat((initPos, targetPos), dim=1)

        aggregatedPredictions = torch.zeros(B, self.modelParams["num_heads"], self.modelParams["future_num_frames"], 2).to(self.device)
        stackedz_l = None
        for i in range(self.modelParams["num_heads"]):
            attn_head = attn_output[:,:, i*head_size:(i + 1)*head_size]
            z_l = torch.cat((tvEncoding, attn_head), dim = 2)
             #out, hidden = self.lstmDec(fullPos, (attn_head, torch.zeros_like(attn_head)))
             #out, hidden = self.lstmDec(attn_head)
            #  hidden, cell = (self.lstmDec).init_hidden_cell(values.shape[1])
            inp = z_l
            if stackedz_l == None:
                stackedz_l = inp
            else:
                stackedz_l = torch.cat((stackedz_l, inp), dim=-1)
            lstmHiddenAndCell = (torch.rand(1, B, self.modelParams["decoder_size"]).to(self.device), torch.rand(1, B, self.modelParams["decoder_size"]).to(self.device))

            # print(inp.shape)
            predictions = torch.zeros(self.modelParams["future_num_frames"], B, 68).to(self.device) #50, 5, 128
            for j in range(self.modelParams["future_num_frames"]):
                output, lstmHiddenAndCell = self.lstmDec(inp, lstmHiddenAndCell)
                output = self.lstmOutputFc(output)
                # print("lstmdec output shape", output.shape)
                #inp = ?
                #_, topi = torch.topk(output,1)
                # topi = (batch*1)
                #input = torch.transpose(topi,0,1)

                predictions[j] = output
                inp = output
                # print('out shape', output.shape)
            predOutput = self.lstmDecFc(predictions) #50, 5, 2
            predOutput = predOutput.permute(1, 0, 2) #5, 50, 4
            aggregatedPredictions[:, i, :, :] = predOutput

        #print(decoded_outputs_t)

        stackedz_l = stackedz_l.squeeze(0) #B, 68*numheads
        #TODO: slight chance that this shouldn't have been a cat, (this generates dependent probs)


        # #generate probabilities 
        probability = self.attn_fc1(stackedz_l)
        probability = self.attn_fc2(probability)
        probability = self.softmax(probability)

        # Move center of predictions back to raster center
        aggregatedPredictions = aggregatedPredictions + self.modelParams['raster_img_center_offset']

        # Transform raster to agent for predictions
        aggregatedPredictions = torch.cat((aggregatedPredictions, torch.ones((B, self.modelParams["num_heads"], self.modelParams["future_num_frames"], 1)).to(self.device)), dim=-1).unsqueeze(-1)
        aggPredRaster = aggregatedPredictions.clone().detach()
        aggregatedPredictionsAgent = torch.matmul(agentFromRaster[:, None, None, :, :], aggregatedPredictions).squeeze(-1)[:, :, :, :2]

        return (aggregatedPredictionsAgent, probability, aggPredRaster)


def rawToRasterScaleSocialState(originalShape, newShape, svPositionsAtT0, svEncoding, lengths):
    '''
    parameters:
        originalShape: starting shape (B, 5+2*Hist, rasterWidth, rasterHeight)
        newShape: resulting shape (B, 512, X, Y) hopefully X, Y = 28, 28
        svPositionsAtT0: tensor with positions to add to grid (B, N, 3)
        svEncoding: tensor with agent data (B, N, LSTMEncoderHidden)
        lengths: Python List of len B, [realAgentsInB_0, realAgentsInB_1...]
    returns:
        social state grid (B, 2, LSTMEncoderHidden)
    '''
    B, _, oldW, oldH = originalShape
    _, _, newW, newH = newShape
    _, _, LSTMEncoderHidden = svEncoding.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = torch.zeros(B, LSTMEncoderHidden, newW, newH).to(device)

    for i in range(svPositionsAtT0.shape[1]):
        lenMask = lengths>i # mask is (B)
        x, y, _ = torch.split(svPositionsAtT0[:, i, :], 1, dim=-1) # x and y should both be (B).  This is the x val of the ith agent, in all batches
        
        svAgentiEncoding = svEncoding[:, i, :] #(B, 1, LSTMEncoderHidden)
        # print("sv Agent i encoding", svAgentiEncoding.shape) #should be (<=B, LSTMEncoderHiddenSize)
        
        xIdx = (x * newW / oldW).long() #(224, 224) -> (28, 28)
        yIdx = (y * newH / oldH).long()

        xMask = (xIdx < newW).squeeze().to(device)
        yMask = (yIdx < newH).squeeze().to(device)
        lenMask = lenMask.to(device)

        # print("un Anded Masks", xMask, yMask, maskT)
        fullMask = torch.logical_and(xMask, yMask).logical_and(lenMask)
        # print("anded Masks", mask)

        xIdx = xIdx[fullMask]
        yIdx = yIdx[fullMask]

        innerMask = []

        for j in range(fullMask.shape[0]):
            # print("maskj: ", mask[j])
            if fullMask[j].item():
                innerMask.append(j)
        # print(innerMask)

        innerMask = torch.tensor(innerMask).long().to(device)
        grid[innerMask, :, xIdx, yIdx] = torch.max(svAgentiEncoding[fullMask], grid[innerMask, :, xIdx, yIdx])

    return grid