"""
train_simple_classifier.py

Очень простой скрипт для дообучения ResNet-18 (freeze backbone + train head)
на задаче бинарной классификации «визитка / не‐визитка».

Usage:
    python train_simple_classifier.py \
        --data_dir /path/to/dataset \
        --epochs 5 \
        --batch_size 32 \
        --lr 1e-3 \
        --output_model card_classifier_simple.pt
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine‐tune ResNet-18 head for business‐card detection (2 classes)."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Путь к корню датасета (папки train/ и val/)."
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Число эпох (рекомендуется 3–7)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size для DataLoader."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Скорость обучения для head‐классификатора."
    )
    parser.add_argument(
        "--output_model", type=str, default="card_classifier_simple.pt",
        help="Куда сохранить обученные веса (.pt)."
    )
    return parser.parse_args()


def train_simple(data_dir: str, epochs: int, batch_size: int, lr: float, output_model: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}\n")

    # ======================
    # 1) Трансформации датасета
    # ======================
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # ======================
    # 2) Создаём ImageFolder
    # ======================
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            data_transforms[x]
        )
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=4,
            pin_memory=True
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    print(f"Классы: {class_names}\n(ожидается ['card', 'not_card'])\n")

    # ======================
    # 3) Загружаем ResNet-18
    # ======================
    # pretrained=True → скачивает веса ImageNet
    model = models.resnet18(pretrained=True)
    # Замораживаем ВСЕ параметры сверточных слоев
    for param in model.parameters():
        param.requires_grad = False

    # Вместо fc (512 → 1000) ставим новый слой (512 → 2)
    num_ftrs = model.fc.in_features  # обычно 512
    model.fc = nn.Linear(num_ftrs, 2)  # 2 класса: card / not_card

    model = model.to(device)

    # Лосс и оптимизатор: только для параметров model.fc
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    best_val_acc = 0.0

    # ======================
    # 4) Цикл обучения
    # ======================
    for epoch in range(epochs):
        print(f"=== Эпоха {epoch+1}/{epochs} ===")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)           # (batch_size, 2)
                    _, preds = torch.max(outputs, 1)  # предсказанные индексы (0 или 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            elapsed = time.time() - start_time

            print(
                f"{phase.capitalize():5s} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Acc: {epoch_acc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Сохраняем лучшую модель по метрике валидности
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), output_model)
                print(f"  🔥 Сохранена лучшая модель: {output_model} (Val Acc = {best_val_acc:.4f})\n")

    print(f"\nОбучение завершено. Best Val Acc: {best_val_acc:.4f}")
    print(f"Весы сохранены в {output_model}")


if __name__ == "__main__":
    args = parse_args()
    train_simple(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_model=args.output_model
    )
