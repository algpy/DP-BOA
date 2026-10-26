# Third-party notices

Original contributions in this release are provided under the Apache License,
Version 2.0; see [LICENSE](LICENSE). Third-party code retains its applicable
copyright and license terms. Dataset and pretrained-weight licenses are separate
from the source-code license.

## DINO and vision transformers

`model/vision_transformer.py` retains the original notice:

> Copyright (c) Facebook, Inc. and its affiliates.

It is licensed under Apache-2.0 and acknowledges its timm implementation source.
The DINO-style model utilities also build on the DINO approach.

- [DINO source](https://github.com/facebookresearch/dino)
- [DINO Apache-2.0 license](https://github.com/facebookresearch/dino/blob/main/LICENSE)
- [timm vision transformer source](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

## Supervised contrastive loss

`model/utils.py` attributes `SupConLoss` to
[SupContrast](https://github.com/HobbitLong/SupContrast). Its BSD-2-Clause copyright,
conditions and disclaimer are reproduced in
[licenses/SupContrast-BSD-2-Clause.txt](licenses/SupContrast-BSD-2-Clause.txt).

> Copyright (c) 2020, Yonglong Tian. All rights reserved.

## Torchvision dataset code

Dataset implementations, including the Food-101 and Oxford-IIIT Pet loaders,
use torchvision code and APIs. The torchvision BSD-3-Clause license is reproduced
in [licenses/torchvision-BSD-3-Clause.txt](licenses/torchvision-BSD-3-Clause.txt).

- [Torchvision source](https://github.com/pytorch/vision)
- [License for torchvision v0.17.1](https://github.com/pytorch/vision/blob/v0.17.1/LICENSE)

Third-party notices are retained in this anonymous release because they describe
upstream rights and attribution, rather than the identity of this paper's authors.
