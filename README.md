
## Language model with adversarial loss

Explorations into adversarial losses on top of autoregressive loss for language modeling

I have tried this in the past, when GANs were still dominant. But at the time I was either too inexperienced or the research not there. Either way could not get it working. Will give it another shot in the next few weeks, mainly to see if an adversarial system could benefit [world modeling](https://github.com/lucidrains/improving-transformers-world-model-for-rl)

## Usage

```python
import torch

from transformer_lm_gan import (
    LanguageModelGenerator,
    Discriminator,
    GAN,
)

gan = GAN(
    strategy = 'gumbel_one_hot', # or 'rotate' for rotation trick, may try combination of two if both fails in experiments
    generator = dict(
        num_tokens = 256,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        max_seq_len = 1024
    ),
    discriminator = dict(
        num_tokens = 256,
        dim = 512,
        depth = 2,
        dim_head = 64,
        heads = 9,
        max_seq_len = 1024
    )
).cuda()

seq = torch.randint(0, 256, (2, 1024)).cuda()

discr_loss = gan.discriminate_forward(seq)
discr_loss.backward()

gen_loss = gan.generate_forward(seq)
gen_loss.backward()
```

## Citations

```bibtex
@inproceedings{Huang2025TheGI,
    title   = {The GAN is dead; long live the GAN! A Modern GAN Baseline},
    author  = {Yiwen Huang and Aaron Gokaslan and Volodymyr Kuleshov and James Tompkin},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:275405495}
}
```

```bibtex
@article{Fifty2024Restructuring,
    title   = {Restructuring Vector Quantization with the Rotation Trick},
    author  = {Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iyengar, Jerry W. Liu, Ehsan Amid, Sebastian Thrun, Christopher Ré},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.06424},
    url     = {https://api.semanticscholar.org/CorpusID:273229218}
}
```
