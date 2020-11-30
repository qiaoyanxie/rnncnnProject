import torch
import torch.nn as nn
import numpy as np
from torchvision.models.resnet import resnet50

def agentToImg(agentsHist, agentsTransf, targetTransf):
    """
    parameters:
        agentsHist (tensor of shape (B, N, H, 3)): tensor of history values of surrounding agents (x, y, theta)
        agentsTransf (tensor of shape (B, N, 4, 4)): tensor of agent - agent to world
        targetTransf (tensor of shape (B, 4, 4)): tensor of target - world to raster
    returns:
        tensor of shape (B, N, H, 3) containing image coordinates of surrounding agents
    """
    # (B, N, 4, 4) Agent to raster batched correctly
    B, N, H, _ = agentsHist.shape
    agentsHistExt = torch.transpose(torch.cat((agentsHist, torch.ones((B, N, H, 1))), dim=3), 2, 3)
    world = torch.matmul(agentsTransf, agentsHistExt)
    return torch.transpose(torch.matmul(targetTransf, world), 2, 3)[:, :, :, :3]

def makeTransformation(pos, scale):
    """
    parameters:
        pos (tensor of shape (B, N, 3)): x, y, theta
    returns:
        tensor of transformation matrix of the form:
            [
                [torch.cos(theta), -np.sin(theta), 0, x],
                [torch.sin(theta), np.cos(theta), 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
    """
    B, N, d = pos.shape
    assert(d == 3)

    x, y, theta = torch.split(pos, 1, dim=2)
    theta = theta.squeeze()
    transf = torch.zeros((B, N, d+1, d+1))
    
    transf[:, :, 2, 2] = 1
    transf[:, :, 3, 3] = 1
    transf[:, :, 0, 0] = scale*torch.cos(theta)
    transf[:, :, 0, 1] = scale*-torch.sin(theta)
    transf[:, :, 1, 0] = scale*torch.sin(theta)
    transf[:, :, 1, 1] = scale*torch.cos(theta)
    transf[:, :, 0, 3] = x.squeeze()
    transf[:, :, 1, 3] = y.squeeze()
    transf[:, :, 2, 3] = theta

    return transf

from l5kit.geometry import transform_points

def testNaman(item):
    """
    parameters:
        item: dict with others_len, others_dict, ego_len, ego_dict etc
    returns
        iteratively compare poss and poss2
    """
    print(item['others_len'][0], item['others_dict'][0]['history_positions'].shape[1], )
    agentsHist = torch.zeros((1, item['others_len'][0], item['others_dict'][0]['history_positions'].shape[1], 3))
    yaw = torch.zeros((item['others_dict'][0]['history_positions'].shape[1], 1))

    # print('agentHist', agentsHist.shape)

    raster_from_world = item['ego_dict']['raster_from_world'][0]
    print("target true raster from world", raster_from_world)
    scale = (raster_from_world[0, 0]**2 + raster_from_world[1, 0]**2)**0.5
    theta = torch.arccos(raster_from_world[0, 0]/scale)
    theta += 2*((np.pi - theta) if raster_from_world[1, 0] < 0 else 0)
    print("calc theta", theta)
    # theta = torch.arccos(raster_from_world[0, 0]/scale)
    posTarget = torch.tensor([raster_from_world[0][2], raster_from_world[1][2], theta]).unsqueeze(0).unsqueeze(0)

    print("cos theta", raster_from_world[0, 0])
    print("sin theta", raster_from_world[1, 0])
    print("raster_from_world[0, 0]/scale", raster_from_world[0, 0]/scale)
    print("dx for target: ", raster_from_world[0, 2])
    print("target yaw", item['ego_dict']['yaw'])
    posAgents = torch.zeros((1, item['others_len'], 3))

    for i in range(item['others_len'][0]):
        yaw[:, 0] = item['others_dict'][i]['history_yaws'].squeeze()
        agentsHist[0, i] = torch.cat((item['others_dict'][i]['history_positions'][0], yaw), dim=-1)

        raster_from_world = item['others_dict'][i]['world_from_agent'][0]
        # print("agent ", i, "true world from agent", raster_from_world)
        posAgents[0, i] = torch.tensor((raster_from_world[0][2], raster_from_world[1][2], item['others_dict'][i]['yaw']))
        # print("agent ", i, " yaw: ", item['others_dict'][i]['yaw'])


    agentsTransf = makeTransformation(posAgents, 1) # agent to world
    targetTransf = makeTransformation(posTarget, scale)[0] #world to raster
    # print("agentsTransf", agentsTransf)
    print("our world to raster: ", targetTransf)
    # print("agentsHist[0, 0]", agentsHist[0,0])
    # print(agentsHist.shape, agentsTransf.shape, targetTransf.shape)
    poss1 = worldToImg(agentsHist, agentsTransf, targetTransf)
    # print("poss1.shape", poss1.shape)
    # pos: B, N, H, 3 (x, y, theta)
    
    for b in range(poss1.shape[0]):
        for i in range(item['others_len'][b]):
            poss2 = transform_points(item['others_dict'][i]['history_positions'][0].float(), item['others_dict'][i]["world_from_agent"][0].float())
            poss3 = transform_points(poss2, item['ego_dict']["raster_from_world"][0].float())
            # poss3 is (H, 2)
            
            # print("world to raster ours: ", poss3)
            if not poss1[b, i, :, 0:2].equal(poss3):
                # print(poss1[b, i, :, 0:2], poss3)
                return False


    return True

