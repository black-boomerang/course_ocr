import datetime
import gc
import os
import time
from collections import defaultdict
from typing import Tuple, Any

import numpy as np
import torch
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from data_reader import Vocabulary
from dataset import init_dataloaders
from model import ConvNeXtWithArcFace, LeNetWithArcFace


def show_plot(history: defaultdict, elapsed_time: int, epoch: int) -> None:
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train']['iters'], history['train']['loss'], label='train')
    plt.plot(history['val']['iters'], history['val']['loss'], label='val')
    plt.ylabel('Лосс', fontsize=15)
    plt.xlabel('Итерация', fontsize=15)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val']['iters'], history['train']['accuracy'], label='train')
    plt.plot(history['val']['iters'], history['val']['accuracy'], label='val')
    plt.ylabel('Точность', fontsize=15)
    plt.xlabel('Итерация', fontsize=15)
    plt.legend()

    plt.suptitle(f'Итерация {history["train"]["iters"][-1]}, эпоха: {epoch}, время: '
                 f'{datetime.timedelta(seconds=elapsed_time)}, лосс: {history["train"]["loss"][-1]:.3f}', fontsize=15)

    plt.show()


def validate(dataloader: DataLoader, model: nn.Module, return_pred: bool = False) -> Tuple:
    model.eval()

    losses = []
    accuracies = []
    preds = []
    for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(config.device)
        y = y.to(config.device)

        embeddings = model(x)
        loss = model.loss_fn(embeddings, y)
        logits = torch.matmul(F.normalize(embeddings), F.normalize(model.loss_fn.W, dim=0))
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        accuracies.append((pred == y.cpu().numpy()).mean().item())
        losses.append(loss.item())

        if return_pred:
            preds.extend(pred)

    if return_pred:
        return np.mean(accuracies), np.mean(losses), preds

    return np.mean(accuracies), np.mean(losses)


def save_state(model: nn.Module, optimizer: Any, scheduler: Any, save_path: str) -> None:
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, save_path)


def load_state(model: nn.Module, optimizer: Any, scheduler: Any, load_path: str) -> None:
    state = torch.load(load_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'], strict=False).to(config.device)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])


def train(train_path: str, lr: float, epochs: int, checkpoint_path: str, lr_decay_rate: float = 0.9,
          embed_dim: int = 128, pretrained_path: str = None, start_epoch: int = 0) -> Tuple[nn.Module, Vocabulary]:
    gc.collect()

    train_dataloader, val_dataloader = init_dataloaders(train_path)
    num_classes = train_dataloader.dataset.helper.vocabulary.num_classes()
    batch_per_epoch = len(train_dataloader)

    # model = LeNetWithArcFace(num_classes, embed_dim).to(config.device)
    model = ConvNeXtWithArcFace(num_classes, embed_dim).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.StepLR(optimizer, batch_per_epoch, gamma=lr_decay_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate, patience=50, min_lr=1e-4)

    if pretrained_path is not None:
        load_state(model, optimizer, scheduler, pretrained_path)

    iteration = 0
    history = defaultdict(lambda: defaultdict(list))
    losses = []
    accuracies = []
    os.makedirs(checkpoint_path, exist_ok=True)
    start_time = time.time()
    for epoch in range(epochs):
        if epoch < start_epoch:
            for _ in range(len(train_dataloader)):
                optimizer.step()
                scheduler.step()
            continue

        model.train()

        pbar = tqdm(enumerate(train_dataloader), total=batch_per_epoch)
        for i, (x, y) in pbar:
            optimizer.zero_grad()

            x = x.to(config.device)
            y = y.to(config.device)

            embeddings = model(x)
            loss = model.loss_fn(embeddings, y)

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                logits = torch.matmul(F.normalize(embeddings), F.normalize(model.loss_fn.W, dim=0))
                pred = torch.argmax(logits, dim=1).cpu().numpy()

            accuracies.append((pred == y.cpu().numpy()).mean().item())
            losses.append(loss.item())
            pbar.set_description(f'Loss: {losses[-1]:.2f}, acc = {accuracies[-1] * 100:.1f}%')
            iteration += 1

        history['train']['accuracy'].append(np.mean(accuracies))
        history['train']['loss'].append(np.mean(losses))
        history['train']['iters'].append(iteration)
        losses = []
        accuracies = []

        accuracy, loss = validate(val_dataloader, model)
        history['val']['accuracy'].append(accuracy)
        history['val']['loss'].append(loss)
        history['val']['iters'].append(iteration)

        clear_output()
        show_plot(history, int(time.time() - start_time), epoch)

        save_state(model, optimizer, scheduler, os.path.join(checkpoint_path, f'epoch{epoch}.pt'))

    save_state(model, optimizer, scheduler, os.path.join(checkpoint_path, f'latest.pt'))

    clear_output()
    show_plot(history, int(time.time() - start_time), epochs)

    return model, train_dataloader.dataset.helper.vocabulary
