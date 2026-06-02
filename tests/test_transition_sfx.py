import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.config import TRANSITION_SFX_DIR
from app.utils.audio import transition_sfx


class TransitionSfxTests(unittest.TestCase):
    def test_default_sfx_dir_points_to_bundled_assets(self):
        self.assertEqual(transition_sfx.SFX_DIR, TRANSITION_SFX_DIR)
        self.assertTrue((transition_sfx.SFX_DIR / "soft_whoosh.mp3").exists())

    def test_default_transition_resolves_real_sfx_file(self):
        resolved = transition_sfx.resolve_transition_sfx("fade")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, "soft_whoosh.mp3")

    def test_default_sfx_gain_is_audible(self):
        resolved = transition_sfx.resolve_transition_sfx("fade")
        self.assertIsNotNone(resolved)
        self.assertGreaterEqual(transition_sfx.sfx_volume_for_path(resolved), 1.2)
        self.assertLessEqual(transition_sfx.sfx_volume_for_path(resolved), 1.3)
        self.assertGreaterEqual(transition_sfx.TRANSITION_SFX_MIX_WEIGHT, 2.0)

    def test_repeated_gentle_transitions_are_spaced_and_varied(self):
        events = [
            {"time": 3.0, "transition": "crossfade"},
            {"time": 6.0, "transition": "fade_eased"},
            {"time": 10.0, "transition": "crossfade"},
            {"time": 14.0, "transition": "fade_eased"},
            {"time": 18.0, "transition": "crossfade"},
            {"time": 23.0, "transition": "fade_eased"},
        ]

        cues = transition_sfx.plan_transition_sfx_cues(events)
        stems = [cue["stem"] for cue in cues]

        self.assertLess(len(cues), len(events))
        self.assertGreater(len(set(stems)), 1)
        self.assertFalse(any(a == b for a, b in zip(stems, stems[1:])))

    def test_impact_hits_have_cooldown(self):
        events = [
            {"time": 10.0, "transition": "zoom_dissolve"},
            {"time": 18.0, "transition": "zoom_dissolve"},
            {"time": 30.0, "transition": "zoom_dissolve"},
        ]

        cues = transition_sfx.plan_transition_sfx_cues(events)
        impact_times = [cue["time"] for cue in cues if cue["stem"] == "impact_hit"]

        self.assertEqual(impact_times, [10.0, 30.0])

    def test_mix_builds_limited_sfx_bed_without_final_limiter(self):
        with TemporaryDirectory() as tmp_dir:
            voice_path = Path(tmp_dir) / "voice.wav"
            output_path = Path(tmp_dir) / "mixed.wav"
            voice_path.write_bytes(b"fake voice")
            captured = {}

            def fake_run(cmd, **_kwargs):
                captured["cmd"] = cmd

                class Result:
                    stderr = ""

                return Result()

            with patch("app.utils.audio.transition_sfx.subprocess.run", side_effect=fake_run):
                result = transition_sfx.mix_transition_sfx(
                    str(voice_path),
                    [
                        {"time": 3.0, "transition": "crossfade"},
                        {"time": 9.0, "transition": "fade_eased"},
                    ],
                    str(output_path),
                    duration=12.0,
                )

        self.assertEqual(result, str(output_path))
        filter_complex = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
        self.assertIn("[sfxbed]", filter_complex)
        self.assertIn("alimiter=limit=", filter_complex)
        self.assertIn("[voice][sfxbed]amix=inputs=2", filter_complex)
        self.assertNotIn("[mix", filter_complex)
        self.assertNotIn("[aout]alimiter", filter_complex)


if __name__ == "__main__":
    unittest.main()
