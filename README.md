# Directed Transformer


- [ ] Build a vanilla transformer model
  - [ ] Text encoding
  - [ ] TensorBoard
  - [ ] Validate performance
  - [ ] Visualize attentions
- [ ] Play around with different decoding schemes
- [ ] Directed Transformer

...

## Preprocess

To create a dataset using existing `train.txt` and `test.txt` files
```bash
python preprocess.py --input train.txt --output-dir /tmp/output --output-name train --target-vocab-size 500 --verbose
python preprocess.py --input test.txt --output-dir /tmp/output --output-name test --verbose

```