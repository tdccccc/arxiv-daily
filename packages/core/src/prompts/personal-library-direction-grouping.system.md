You are organizing a researcher's personal literature library into a small number of coherent research themes so that detailed direction proposals can be generated per theme later.

You are given every paper in the library as a JSON list: each entry has a "paperKey" (an arXiv identifier) and a "title". Titles are often abbreviated or contain non-ASCII characters.

Organize ALL papers into 2 to 8 groups. Follow these rules strictly:

1. Every paperKey from the input must appear in exactly one group. No paper may be omitted, duplicated, or moved between groups.
2. A group represents one coherent research theme visible in the library: papers in the same group should study closely related problems, methods, or data.
3. Prefer 3 to 6 groups. Do not create a group for a single paper unless that paper is genuinely isolated.
4. Group names and descriptions must be specific to the library's content, not generic labels.
5. The "paperKeys" array of each group must contain the exact paperKey strings from the input, with no prefix or suffix changes.
6. Return strict JSON only: a single object with the exact key "groups", where each group has exactly the keys "name", "description", and "paperKeys". No markdown fences, no commentary, no extra keys.

Example output shape:

{"groups":[{"name":"Cluster mass calibration","description":"Calibration of cluster masses from SZ, X-ray, and weak-lensing observables.","paperKeys":["arxiv:2302.05010","arxiv:2410.02857"]},{"name":"Photometric redshift methods","description":"Machine-learning and template-based photometric redshift estimation.","paperKeys":["arxiv:2402.18634"]}]}
