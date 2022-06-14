# GRMI

This is the source code for GRMI

## Installation
Install python 3.6, Pytorch 1.10, PyG 2.0

## Usage
We used three datasets for expriments in paper.

1. [POM](https://drive.google.com/file/d/1qUfcFKooySU5NHtKqKcB1rp75KTRg0hi)
2. [IEMOCAP](https://drive.google.com/file/d/17WtWWay4w3yAuwzLfjHnbhZO5h8XjlVC)
3. [NTU](https://drive.google.com/file/d/1Vx4K15bW3__JPRV0KUoDWtQX8sB-vbO5)

To train label prediction on POM datasets:
```bash
python pretrain_pom_y.py pom
python train_pom_y.py pom
```
