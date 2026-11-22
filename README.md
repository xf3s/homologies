# homologies

What do different language models grow in common, and why? Small pre-registered experiments by **three**, with **umber** (Claude Opus 5.5, a Claude Code resident in three's household).

In biology a **homology** is a part shared by descent (your arm bone and a bat's wing bone). An **analogy** is a part grown separately for the same job (a bird's wing and a bee's). Language models have both: a model trained further from another one keeps its neurons, and models that share no ancestry still grow alike, because they eat the same diet of human text. These pages try to tell the two apart, and to ask what passes from one model to the next when models learn from each other's writing.

three picks the questions and asks the follow-ups. umber writes the code, runs it, and explains the results. Everything here runs on a laptop and a small home server, with no GPU.

**How we work.** Before each test runs, we write down what we expect it to show, in a dated lab logbook, together with a check that could break the result. Each page says which predictions held.

## The findings

| | what we found, in one line |
|---|---|
| [01 · groups, not neurons](01-groups-not-neurons/) | Three Pythia-160m runs from different seeds: 3% of neurons have a twin, 63–70% sit in groups that have a partner. Convergence, not inheritance |

Sister repos: [cyborg-peer-review](https://github.com/xf3s/cyborg-peer-review) (Llama-3.1-8B) and [little-qwen-field-notes](https://github.com/xf3s/little-qwen-field-notes) (Qwen3.5-0.8B).

## License

Writing, pictures and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please credit **three** (with umber). Code: MIT (see `LICENSE`).
