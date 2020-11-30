import bisect
import os
from copy import deepcopy
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.rasterization import StubRasterizer, build_rasterizer
from torch.utils.data import DataLoader, Dataset, Subset

# is_kaggle = os.path.isdir("/kaggle")


# data_root = (
#     "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
#     if is_kaggle
#     else "lyft-motion-prediction-autonomous-vehicles"
# )

data_root = ("/content/lyftdataset")

CONFIG_DATA = {
    "format_version": 4,
    "model_params": {
        "model_architecture": "resnet50",
        "history_num_frames": 10,
        "history_step_size": 1,
        "history_delta_time": 0.1,
        "future_num_frames": 50,
        "future_step_size": 1,
        "future_delta_time": 0.1,
    },
    "raster_params": {
        "raster_size": [256, 256],
        "pixel_size": [0.5, 0.5],
        "ego_center": [0.25, 0.5],
        "map_type": "py_semantic",
        "satellite_map_key": "aerial_map/aerial_map.png",
        "semantic_map_key": "semantic_map/semantic_map.pb",
        "dataset_meta_key": "meta.json",
        "filter_agents_threshold": 0,
        "disable_traffic_light_faces": False,
    },
    "train_dataloader": {
        "key": "scenes/sample.zarr",
        "batch_size": 24,
        "shuffle": True,
        "num_workers": 0,
    },
    "val_dataloader": {
        "key": "scenes/validate.zarr",
        "batch_size": 24,
        "shuffle": False,
        "num_workers": 4,
    },
    "test_dataloader": {
        "key": "scenes/test.zarr",
        "batch_size": 24,
        "shuffle": False,
        "num_workers": 4,
    },
    "train_params": {
        "max_num_steps": 400,
        "eval_every_n_steps": 50,
    },
}


class MultiAgentDataset(Dataset):
    def __init__(
        self,
        rast_only_agent_dataset: AgentDataset,
        history_agent_dataset: AgentDataset,
    ):
        super().__init__()
        self.rast_only_agent_dataset = rast_only_agent_dataset
        self.history_agent_dataset = history_agent_dataset

    def __len__(self) -> int:
        return len(self.rast_only_agent_dataset)

    def get_others_dict(
        self, index: int, ego_dict: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        agent_index = self.rast_only_agent_dataset.agents_indices[index]
        frame_index = bisect.bisect_right(
            self.rast_only_agent_dataset.cumulative_sizes_agents, agent_index
        )
        frame_indices = self.rast_only_agent_dataset.get_frame_indices(frame_index)
        assert len(frame_indices) >= 1, frame_indices
        frame_indices = frame_indices[frame_indices != index]

        historyCount = ego_dict['history_positions'].shape[0]

        # 3 is for x, y, theta
        othersIdx = 0
        othersHist = torch.zeros((len(frame_indices), historyCount, 3))
        othersPos = torch.zeros((len(frame_indices), 3))
        # The centroid of the AV in the current frame in world reference system. Unit is meters
        for idx, agent in zip(  # type: ignore
            frame_indices,
            Subset(self.history_agent_dataset, frame_indices),
        ):
            world_from_agent = torch.tensor(agent['world_from_agent'])
            theta, _ = calculateTheta(world_from_agent)

            othersPos[othersIdx] = torch.tensor([world_from_agent[0][2], world_from_agent[1][2], theta])
            othersHist[othersIdx] = torch.cat((torch.tensor(agent['history_positions']), torch.tensor(agent['history_yaws'])), dim=-1)

            othersIdx += 1
    
        return othersHist, othersPos


    def __getitem__(self, index: int) -> Dict[str, Any]:
        rast_dict = self.rast_only_agent_dataset[index]
        ego_dict = self.history_agent_dataset[index]

        # others_dict, others_len = self.get_others_dict(index, ego_dict, historyCount)
        othersHist, othersPos = self.get_others_dict(index, ego_dict)

        targetHist = torch.cat((torch.tensor(ego_dict['history_positions']), torch.tensor(ego_dict['history_yaws'])), dim=-1).unsqueeze(0)
        agentsHist = torch.cat((targetHist, othersHist), dim=0) # (1+N, H, 3)
        
        world_from_agent = torch.tensor(ego_dict['world_from_agent'])
        theta, _ = calculateTheta(world_from_agent)
        targetPos = torch.tensor([world_from_agent[0][2], world_from_agent[1][2], theta]).unsqueeze(0)
        agentsPos = torch.cat((targetPos, othersPos), dim=0) # (1+N, 3)

        raster_from_world = torch.tensor(ego_dict['raster_from_world'])
        theta, scale = calculateTheta(raster_from_world)
        targetPos = torch.tensor([raster_from_world[0][2], raster_from_world[1][2], theta]).unsqueeze(0)

        agentsTransf = makeTransformation(agentsPos, 1) # agent to world
        targetTransf = makeTransformation(targetPos, scale).squeeze(0)

        agentsHist = agentToImg(agentsHist, agentsTransf, targetTransf)

        raster_from_agent = torch.tensor(ego_dict['raster_from_agent'], dtype=torch.float)
        targetPos = torch.tensor(ego_dict['target_positions'])
        targetPos = torch.transpose(torch.cat((targetPos, torch.ones((targetPos.shape[0], 1))), dim=-1), 0, 1)
        targetPos = torch.transpose(torch.matmul(raster_from_agent, targetPos), 0, 1)[:, :2]

        agent_from_raster = torch.inverse(raster_from_agent)

        return (agentsHist, agentsHist.shape[0], torch.tensor(rast_dict["image"]), targetPos, torch.tensor(ego_dict['target_availabilities']), agent_from_raster)

def calculateTheta(transf):
    scale = (transf[0, 0]**2 + transf[1, 0]**2)**0.5
    theta = torch.arccos(transf[0, 0]/scale)
    return theta + 2*((np.pi - theta) if transf[1, 0] < 0 else 0), scale

def agentToImg(agentsHist, agentsTransf, targetTransf):
    """
    parameters:
        agentsHist (tensor of shape (N, H, 3)): tensor of history values of surrounding agents (x, y, theta)
        agentsTransf (tensor of shape (N, 4, 4)): tensor of agent - agent to world
        targetTransf (tensor of shape (4, 4)): tensor of target - world to raster
    returns:
        tensor of shape (N, H, 3) containing image coordinates of surrounding agents
    """
    N, H, _ = agentsHist.shape
    agentsHistExt = torch.transpose(torch.cat((agentsHist, torch.ones((N, H, 1))), dim=2), 1, 2)
    world = torch.matmul(agentsTransf, agentsHistExt)
    return torch.transpose(torch.matmul(targetTransf, world), 1, 2)[:, :, :3]

def makeTransformation(pos, scale):
    """
    parameters:
        pos (tensor of shape (N, 3)): x, y, theta
    returns:
        tensor of transformation matrix of the form (N, 4, 4):
            [
                [torch.cos(theta), -np.sin(theta), 0, x],
                [torch.sin(theta), np.cos(theta), 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
    """
    N, d = pos.shape
    assert(d == 3)

    x, y, theta = torch.split(pos, 1, dim=1)
    theta = theta.squeeze()
    transf = torch.zeros((N, d+1, d+1))
    
    transf[:, 2, 2] = 1
    transf[:, 3, 3] = 1
    transf[:, 0, 0] = scale*torch.cos(theta)
    transf[:, 0, 1] = scale*-torch.sin(theta)
    transf[:, 1, 0] = scale*torch.sin(theta)
    transf[:, 1, 1] = scale*torch.cos(theta)
    transf[:, 0, 3] = x.squeeze()
    transf[:, 1, 3] = y.squeeze()
    transf[:, 2, 3] = theta

    return transf

def collate_fn(data):
    """
    Returns:
        (agentsHists, lengths, rasterImgs, targetPos, targetValid)
    """
    agentsHists, lengths, rasterImgs, targetPos, targetValid, agent_from_raster = zip(*data)
    agentsHists = torch.nn.utils.rnn.pad_sequence(agentsHists, batch_first=True)
    lengths = torch.tensor(lengths)
    rasterImgs = torch.stack(rasterImgs)
    targetPos = torch.stack(targetPos)
    targetValid = torch.stack(targetValid)
    agentFromRaster = torch.stack(agent_from_raster)

    return (agentsHists, lengths, rasterImgs, targetPos, targetValid, agentFromRaster)


class LyftAgentDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict = CONFIG_DATA, data_root: str = data_root):
        super().__init__()
        self.cfg = cfg
        self.dm = LocalDataManager(data_root)
        self.rast = build_rasterizer(self.cfg, self.dm)

    def chunked_dataset(self, key: str):
        dl_cfg = self.cfg[key]
        dataset_path = self.dm.require(dl_cfg["key"])
        zarr_dataset = ChunkedDataset(dataset_path)
        zarr_dataset.open()
        return zarr_dataset

    def get_dataloader_by_key(
        self, key: str, mask: Optional[np.ndarray] = None
    ) -> DataLoader:
        dl_cfg = self.cfg[key]
        zarr_dataset = self.chunked_dataset(key)
        agent_dataset = AgentDataset(
            self.cfg, zarr_dataset, self.rast, agents_mask=mask
        )
        return DataLoader(
            agent_dataset,
            shuffle=dl_cfg["shuffle"],
            batch_size=dl_cfg["batch_size"],
            num_workers=dl_cfg["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def train_dataloader(self):
        key = "train_dataloader"
        return self.get_dataloader_by_key(key)

    def val_dataloader(self):
        key = "val_dataloader"
        return self.get_dataloader_by_key(key)

    def test_dataloader(self):
        key = "test_dataloader"
        test_mask = np.load(f"{data_root}/scenes/mask.npz")["arr_0"]
        return self.get_dataloader_by_key(key, mask=test_mask)


class MultiAgentDataModule(LyftAgentDataModule):
    def __init__(self, cfg: Dict = CONFIG_DATA, data_root: str = data_root) -> None:
        super().__init__(cfg=cfg, data_root=data_root)
        stub_cfg = deepcopy(self.cfg)

        # this can be removed once l5kit supporting passing None as rasterizer
        # https://github.com/lyft/l5kit/pull/176
        stub_cfg["raster_params"]["map_type"] = "stub_debug"
        self.stub_rast = build_rasterizer(stub_cfg, self.dm)
        assert isinstance(self.stub_rast, StubRasterizer)

    def get_dataloader_by_key(
        self, key: str, mask: Optional[np.ndarray] = None
    ) -> DataLoader:
        dl_cfg = self.cfg[key]
        zarr_dataset = self.chunked_dataset(key)
        # for the rast only agent dataset, we'll disable rasterization for history frames, so the
        # channel size will only be 3 + 2 (for current frame)
        no_history_cfg = deepcopy(self.cfg)
        no_history_cfg["model_params"]["history_num_frames"] = 0
        rast_only_agent_dataset = AgentDataset(
            no_history_cfg, zarr_dataset, self.rast, agents_mask=mask
        )
        history_agent_dataset = AgentDataset(
            self.cfg, zarr_dataset, self.stub_rast, agents_mask=mask
        )
        return DataLoader(
            MultiAgentDataset(
                rast_only_agent_dataset=rast_only_agent_dataset,
                history_agent_dataset=history_agent_dataset
            ),
            shuffle=dl_cfg["shuffle"],
            batch_size=dl_cfg["batch_size"],
            num_workers=dl_cfg["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )


