# SSEAL Self-Supervised Explorative Agent Learning
**This is the project repository for the SSEAL project for UC Berkeley's CS294-196 Fall 2024**

Authors: Evan Frick, Dylan Goetting, Dhruv Gautam, Arjun Kohli, Keshav Singhal

## Setup

### Install Python Packages

`pip install -r requirements.txt`

### Environment

Populate a `.env` file in the root directory of the project with the following keys:

```
FIREWORKS_API_KEY=<key3>
ANTHROPIC_API_KEY=<key4>
OPENAI_API_KEY=<key5>
```

## Example Usage

```
python -m src.main \
  --model, -m # Which model is used for the execution agent
  --model-type, -mt # This model's provider (eg. openai)
  --explore-model, -em # The model use for exploration
  --explore-model-type, emt  # The explore model's provider (eg. openai)
  --execute-temp, -et # What temp to execute at (default 0)
  --explore-temp, -ept # What tempt to explore at (default 0)
  --explore-environment-iterations, -eei # How many iterations of SSEAL to run.
  --max_iterations_per_episode, -mipe # How many iterations per query
  --benchmark_path, -bp # Which benchmark to run, should be a .py file in src/benchmarks
  --task, -t # Which task to run (eg. sports_data)
```

### Example Command

Running gpt-4o-mini as agent with gpt-4o as the exploration model.

```
python -m src.main -bp linux_terminal.py --task linux_terminal -mipe 10 -eei 4 -m gpt-4o-mini -em gpt-4o
```

### Caching

To save time, the exploration phase is cached in `caches/`. The cache is specific to a task, # of exporation iterations, and model. You can clear a cache by deleting it. The repo comes pre-loaded with caches for our provided test tasks.

### Optimizing a Custom API

Let's say we have a py file containing our new function context. We first should put this into `src/benchmarks/`. For example, we can see a `math_demo.py`. Then we would run: `python -m src.main -bp math_demo.py --task custom -mipe 0 -eei <num_exporation> -m gpt-4o -mt openai`. In this case, `-mipe 0` indicates that we don't want to execute any testing tasks, but just want to run prompt optimization. We could see the output of the exploration either in the log file generated ([logs/](logs/)) or get the metadata in [cache/](cache/). It will be located in `cache/<function_file_name>/<explore_model_name>.json`. In particular, it will be under the key `<num_exporation>`.

### Experiment Commands

The commands we ran for this project are in `commands.sh`. They can be run with `source commands.sh`. Running all of them will probably take many days. See [analysis.ipynb](analysis.ipynb) for generating the graphs in our paper. We also provide the outputs from our experiments in [experiments/](/experiments/).
