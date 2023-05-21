## PointDistiller based on MMdet3D

#### Step-1: Install mmdetection3D
* Install PointDistiller based on mmdetection3d [instruction](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).
  - Pytorch==1.10.0+cu113
  - Torchviison==0.11.0+cu113
  - mmdet==2.22.0
  - mmcv-full==1.4.8
  - mmdet3d==1.0.0rc0 # this version
  - setuptools==59.5.0 # required by distutils

#### Step-2: Add KD training to mmdetection3D

Replace their codes with our codes, including *tools/train.py, mmdet3d/apis/train.py, mmdet3d/models/detectors, configs/pointpillars, /anaconda3/lib/python3.8/site-packages/mmcv/runner/base_runner.py & epoch_based_runner.py, anaconda3/lib/python3.8/site-packages/mmdet/models/detectors/base.py, anaconda3/lib/python3.8/site-packages/mmdet/apis/train.py*

* Modifying mmdetection and mmcv to support knowledge distillation:
1. mmdet/apis/train.py:
Add `teacher` parameter to the `train_dector` function:
```python
def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None,
                   teacher=None): # add this line
```
Load teacher model:
```python
# for distributed training
teacher = MMDistributedDataParallel(
            teacher.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
# for single machine
teacher = MMDataParallel(
            teacher.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
```
Add `teacher` parameter to the runner:
```python
runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
            teacher=teacher)) # add this line
```
2. mmcv/runner/base_runner.py & mmcv/runner/epoch_based_runner.py
Add `teacher` parameter to the mmcv `BaseRunner`
```python
def __init__(self,
             model,
             batch_processor=None,
             optimizer=None,
             work_dir=None,
             logger=None,
             meta=None,
             max_iters=None,
             max_epochs=None,
             teacher=None): # add this line
    self.teacher = teacher # add this line
```
Modify mmcv `EpochBasedRunner`:
```python
@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            if self.teacher is not None:
                with torch.no_grad():
                    t_info = self.teacher.train_step(data_batch, self.optimizer, epoch=self.epoch,
                                                iter=self._inner_iter, is_teacher=True, teacher_info=None,
                                                **kwargs)
                outputs = self.model.train_step(data_batch, self.optimizer, epoch=self.epoch, 
                                                iter=self._inner_iter, is_teacher=False, teacher_info=t_info,
                                                **kwargs)
            else:
                outputs = self.model.train_step(data_batch, self.optimizer,
                                                **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
```
3. mmdet/models/dectors/base.py
Add teacher model information:
```python
def train_step(self, data, optimizer, epoch, iter, is_teacher=False, teacher_info=None):
    if is_teacher:
        t_info = self.get_teacher_info(**data)
        return t_info
    else:
        losses = self(**data, teacher_info=teacher_info)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
```

Please make sure that you can run their test exmaples, and then go to step-3.

#### Step-3: Setup KD Teachers

Replace their codes with our codes, including *tools/train.py, mmdet3d/apis/train.py, mmdet3d/models/detectors, configs/pointpillars, /anaconda3/lib/python3.8/site-packages/mmcv/runner/base_runner.py & epoch_based_runner.py, anaconda3/lib/python3.8/site-packages/mmdet/models/detectors/base.py, anaconda3/lib/python3.8/site-packages/mmdet/apis/train.py*

Download the official teacher models from Openmmlab:
```bash
mkdir pretrain && cd pretrain/
wget -c https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth
wget -c https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
wget -c https://download.openmmlab.com/mmdetection3d/v1.0.0_models/point_rcnn/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth
wget -c https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fp16/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth
```
and put it in mmdetection3d/pretrain. You need to create this folder by yourself.

#### Some Tips
- Please make sure that the detectors are trained with the loss `loss_kd` printed, then you should be all set
- The codes are tested on Ubuntu 18.04 with CUDA 11.3 for A100 devices and CUDA 10.2 for RTX 2080Ti devices. We have run 4x compression PointPillars and 4x compression SECOND (`kneighbours=256`) experiments with A100 GPU devices, and we run other PointPillars, SECOND and PointRCNN experiments with 2080Ti GPU devices, which we find a better choice for the performance with current hyper params
- Please change the kd layers and hyper params in the detector files for each model before running
- To better reproduce the results, please take a look at the logs and follow the training configs like GPU device and the number of gpus used, and other training hyper params. We provide training logs and PointPillars student model on [Google Drive](https://drive.google.com/drive/folders/1jA14eMk-0fIywFxku-ijfMIBzFTp7Lef?usp=share_link)
- For PointPillars, we train a network for Car and a network for pedestrians and cyclists following the original paper
- We notice that the pretrained models from official [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) have been updated. It may be helpful when using better teachers, but we have not tried yet

Contact me (`runpei.dong@outlook.com`) if you meet some issues that needs some help.