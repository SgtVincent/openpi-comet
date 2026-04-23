# Installation (Conda Environment)

## 1) Create/Update the conda environment

From the repository root:

```bash
cd openpi-comet

# Create the environment (first time)
mamba env create -f environment.yml

# Or: update the environment (if it already exists)
mamba env update -f environment.yml
```

Activate the environment (the name comes from the `name:` field in the yml):

```bash
conda activate openpi-comet-nas
```

## 2) Install openpi-comet and BEHAVIOR-1K dependencies (Conda)

### 2.1 Install openpi-comet (editable)

Following the intent of the README installation flow (but without `uv sync` and `.venv`), run the following after activating the conda environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cd $PATH_TO_BEHAVIOR_1K

uv pip install -e bddl3
uv pip install -e "OmniGibson[eval]"
```

## 3) Apply the transformers patch (Conda)

The PyTorch implementation in openpi requires patching transformers. 
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* $CONDA_PREFIX/lib/python3.11/site-packages/transformers/
```

Verify transformers still imports:

```bash
python -c "import transformers; print('transformers import ok:', transformers.__version__)"
```
