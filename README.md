
### Load PyTorch models

```bash
python scripts/download_pretrained_models.py
``` 

### The following command executes a dropout algorithm on a 224x224 central crop of an image

```bash
python executable/analysis.py --algorithm x
# where x is either "b", "n", or "p", or with more verbosity "bisection_dropout", "neural_dropout", or "pixel_dropout"
```

### See app/app.py for information on required vs. optional arguments, including

```yaml
algorithm:          "Analysis method (p for pixel_dropout, b for bisection_dropout, n for neural_dropout)."
model-name:         "Model architecture"
image-name:         "Image to analyse"
x:                  "Number of columns (max 224)"
y:                  "Number of rows (max 224)"
num-ouputs:         "Top classifications"
df-output:          "CSV filepath"
img-output:         "Image output"
device:             "cuda or cpu"
device-id:          "GPU index"
n:                  "Number of pixels displayed"
output-folder:      "Filepath where quantitative outputs are stored"
imagenet-classes:   "Filepath where imagenet class data is stored"
injection-clount:   "Number of random injections to introduce (to neural dropout)"
```

