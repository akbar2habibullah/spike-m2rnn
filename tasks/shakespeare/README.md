# Shakespeare — char-LM smoke test (Stage 0)

This is the **smoke test, not the goal** (CLAUDE.md guardrail #7). It proves the ES
+ spiking + matrix-recurrence pipeline learns end-to-end; it does **not** test
state-tracking.

## Data
Drop tiny-shakespeare `input.txt` next to where you run, or pass `data_path`:
```bash
# from src/
python -m spiking_m2rnn.train          # expects ./input.txt
```
nanoGPT convention: full char vocab (~65), 90/10 split. Loader: `spiking_m2rnn.data.load_data`.

## Expected result
`logs/Stage_0.log`: val 5.90 → ~4.23 bpc, firing self-stabilizing ~33%. It will
**not** approach Adam nanoGPT (~1.4 bpc) — expected for ES at this scale.
