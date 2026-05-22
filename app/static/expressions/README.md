# Expression overlay PNGs

Place transparent PNG character renders here (e.g. from Vivre Card Drive sync).

## Expected filenames (fallback)

| File | Emotion |
|------|---------|
| `neutral.png` | calm narration |
| `serious.png` | lore / facts |
| `happy.png` | upbeat lines |
| `excited.png` | hype |
| `angry.png` | conflict |
| `surprised.png` | twists |
| `sad.png` | emotional beats |
| `worried.png` | tension |
| `smirking.png` | sarcasm |
| `confident.png` | bold claims |
| `intense.png` | climax |
| `embarrassed.png` | light humor |

Set `VIVRE_CARD_ASSETS_DIR` in `.env` to a local Google Drive folder for per-character art; the app indexes PNGs automatically.

Entry animations are emotion-based (`pop_in`, `shake_in`, `fade_rise`, etc.) — see `app/utils/expression_effects.py`.
