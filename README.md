An implementation of [Andrej Karpathy's exercise](https://github.com/karpathy/minbpe/blob/master/exercise.md) on building a GPT-4 tokenizer. Implemented in Rust.

On my system, libpython3.12.so is not in the default linker path. When using conda, it can be included from the venv using the following command.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib
```