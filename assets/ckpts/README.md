From the project root, download the public checkpoint files into this folder:

```bash
hf download Louis0324/StyleStream \
  stylizer-no-style-enc.ckpt destylizer.ckpt vocos_causal_best.ckpt \
  --repo-type model --local-dir assets/ckpts
```

After downloading, this folder should contain:

- `stylizer-no-style-enc.ckpt`
- `destylizer.ckpt`
- `vocos_causal_best.ckpt`

If you run the download command from another directory, move these files into `assets/ckpts/`.
