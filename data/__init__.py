import torch.utils.data
import threading


def CreateDataset(opt):
    """loads dataset class"""

    if opt.arch == 'vae' or opt.arch == 'gan':
        if opt.dataset_type == "acronym":
            from data.grasp_sampling_data_acronym import GraspSamplingData
        else:
            from data.grasp_sampling_data_6dof import GraspSamplingData
        dataset = GraspSamplingData(opt)
    else:
        if opt.dataset_type == "acronym":
            from data.grasp_evaluator_data_acronym import GraspEvaluatorData
        else:
            from data.grasp_evaluator_data_6dof import GraspEvaluatorData
        dataset = GraspEvaluatorData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        if opt.dataset_type == "acronym":
            from data.base_dataset_acronym import collate_fn
        else:
            from data.base_dataset_6dof import collate_fn
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.num_objects_per_batch,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
