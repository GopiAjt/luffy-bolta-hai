# Expression overlay PNGs

**Final video expression overlays use only this folder** — not Vivre Card characters.

Place transparent PNGs named by emotion:

| File | Use when narration is… |
|------|-------------------------|
| `neutral.png` | calm, default narration |
| `serious.png` | lore, facts |
| `happy.png` | upbeat |
| `excited.png` | hype |
| `angry.png` | conflict |
| `surprised.png` | twists |
| `sad.png` | emotional beats |
| `worried.png` | tension |
| `smirking.png` | sarcasm |
| `confident.png` | bold claims |
| `intense.png` | climax |
| `embarrassed.png` | light humor |

Gemini may return labels like `emotional` — they map to these files automatically (e.g. `emotional` → `sad.png`).

Entry animations are emotion-based (`pop_in`, `shake_in`, `fade_rise`, etc.) — see `app/utils/expression_effects.py`.

## Vivre Card (image slides only)

B-roll / slide suggestions still use `app/data/vivre-card/` (Characters, Symbols, misc). That does **not** affect face overlays.

## Legacy: Vivre faces on overlays

To restore per-character Vivre overlays (old behavior), set in `.env`:

```bash
EXPRESSION_ASSETS_SOURCE=vivre
```
