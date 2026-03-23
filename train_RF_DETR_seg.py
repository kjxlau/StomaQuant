import os
import sys
import json
import time
import datetime
from collections import defaultdict, deque

import torch
from PIL import Image

# 1. FIX: Import the explicit segmentation class from the rfdetr package
from rfdetr import RFDETRSegMedium
from rfdetr.util.misc import NestedTensor, reduce_dict

# Load Medium segmentation model (COCO pretrained)
model = RFDETRSegMedium()

train_dir_images = "./train" 

coco_dataset = {
    "images":[],
    "annotations":[], # Reminder: Populate this with bounding boxes / segmentation masks!
    "categories":[]
}

annotation_id = 0
image_id_counter = 0

if os.path.exists(train_dir_images):
    for image_fol in os.listdir(train_dir_images):
        image_path = os.path.join(train_dir_images, image_fol)
        
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        
                image_dict = {
                    "id": image_id_counter,
                    "width": width,
                    "height": height,
                    "file_name": image_fol,
                }
        
                coco_dataset["images"].append(image_dict)
                image_id_counter += 1 
                
        except IOError:
            print(f"Skipping non-image file: {image_fol}")
            continue
            
    # Write the dataset to a JSON file so model.train() can find it
    annotation_file = os.path.join(train_dir_images, "_annotations.coco.json")
    with open(annotation_file, "w") as f:
        json.dump(coco_dataset, f)
    print(f"Saved custom COCO annotations to {annotation_file}")
else:
    print(f"Error: Directory not found at path: {train_dir_images}")

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values."""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    @property
    def global_avg(self):
        return self.total / self.count
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=max(self.deque) if len(self.deque) > 0 else 0,
            value=self.deque[-1] if len(self.deque) > 0 else 0)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            if hasattr(meter, 'synchronize_between_processes'):
                 meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header: header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}'])
        if torch.cuda.is_available(): log_msg += ' max mem: {memory:.0f}'
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()

def train_one_epoch_patched(model: torch.nn.Module, criterion: torch.nn.Module,
                    lr_scheduler, data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, batch_size: int, max_norm: float = 0,
                    ema_m=None, schedules=None,
                    num_training_steps_per_epoch=None, vit_encoder_num_layers=None,
                    args=None, callbacks={}):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    use_amp = args is not None and getattr(args, 'amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets =[{k: v.to(device) for k, v in t.items()} for t in targets]

        cloned_tensors = samples.tensors.clone()
        samples = NestedTensor(cloned_tensors, samples.mask)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced_scaled = sum(v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict)
        loss_value = losses_reduced_scaled.item()

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        if ema_m is not None:
            ema_m.update(model)

        metric_logger.update(loss=loss_value, **{k: v for k, v in loss_dict_reduced.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# --- SAFE DYNAMIC MONKEY PATCHING ---
try:
    import rfdetr.engine
    rfdetr.engine.train_one_epoch = train_one_epoch_patched
except ImportError:
    pass # Will be handled by the dynamic loop below

# Iterate loaded modules and dynamically override train_one_epoch wherever it was already imported
for mod_name, mod in list(sys.modules.items()):
    if mod_name.startswith('rfdetr.') and hasattr(mod, 'train_one_epoch'):
        setattr(mod, 'train_one_epoch', train_one_epoch_patched)

history =[]

def callback2(data):
    history.append(data)

if not hasattr(model, "callbacks"):
    model.callbacks = defaultdict(list)
    
model.callbacks["on_fit_epoch_end"].append(callback2)

# specify the no of epochs and batch size for training
model.train(dataset_dir="./train", epochs=300, batch_size=4, lr=1e-4)
