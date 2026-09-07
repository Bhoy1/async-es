# Third-Party Notices

## ES engine utilities

`distributed_utils.py` and `weight_update_utils.py` are derived from code in
[`es-awd`](https://github.com/kschweig/es-awd), which in turn builds on
[`es-at-scale`](https://github.com/VsonicV/es-at-scale). Those projects use an
Academic Public License that permits noncommercial academic, educational, and
nonprofit research use; commercial use requires a separate license. The
applicable license text is included in `LICENSE.txt`.

The derived files were modified in September 2026 to support bounded-staleness
asynchronous ES, deterministic update replay, selectable perturbation scopes,
newer vLLM interfaces, and portable Ray startup.

## Endless Terminals

The Endless Terminals task adapter integrates with the external
[`endless-terminals`](https://github.com/kanishkg/endless-terminals) project,
which is distributed under the Apache License 2.0. This repository does not
vendor its environment implementation or task images; users provide a pinned
checkout and generated task data at runtime.
