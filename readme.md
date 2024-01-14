```
python -m venv venv

pip install datasets
python main.py
```

output dataset is in `./oasst2_guanaco`

load it like this:
```
from datasets import load_from disk
ds = load_from_disk('oasst2_guanaco')
