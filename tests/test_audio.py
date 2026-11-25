"""Post-processing in serve_tts: WSOLA time-stretch and BS.1770 loudness normalization."""

import numpy as np
import pytest
from conftest import tone

import serve_tts

RATE = 24000


def dominant_frequency(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return np.fft.rfftfreq(samples.size, 1 / rate)[np.argmax(spectrum)]


def test_time_stretch_default_speed_is_identity():
    samples = tone()
    assert serve_tts.time_stretch(samples, serve_tts.SPEED_DEFAULT, RATE) is samples


def test_time_stretch_empty():
    empty = np.zeros(0, dtype=np.float32)
    assert serve_tts.time_stretch(empty, 1.5, RATE).size == 0


@pytest.mark.parametrize("speed", [serve_tts.SPEED_MIN, 0.8, 1.25, serve_tts.SPEED_MAX])
def test_time_stretch_changes_duration_and_keeps_pitch(speed):
    samples = tone(seconds=2.0, freq=180.0)
    stretched = serve_tts.time_stretch(samples, speed, RATE)
    assert stretched.dtype == np.float32
    assert stretched.size == round(samples.size / speed)
    assert dominant_frequency(stretched, RATE) == pytest.approx(180.0, abs=2.0)
    # WSOLA overlaps aligned frames, so a steady tone keeps its level instead of beating
    inner = stretched[RATE // 10 : -RATE // 10]
    assert np.abs(inner).max() == pytest.approx(0.1, rel=0.1)


def test_integrated_loudness_of_full_scale_sine():
    # BS.1770: a 997 Hz sine at 0 dBFS reads -3.01 LUFS in one channel
    samples = tone(seconds=3.0, rate=48000, freq=997.0, amplitude=1.0)
    assert serve_tts.integrated_loudness(samples, 48000) == pytest.approx(-3.01, abs=0.05)


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_integrated_loudness_scales_with_gain(rate):
    loud = serve_tts.integrated_loudness(tone(seconds=2.0, rate=rate, freq=997.0, amplitude=0.5), rate)
    quiet = serve_tts.integrated_loudness(tone(seconds=2.0, rate=rate, freq=997.0, amplitude=0.05), rate)
    assert loud - quiet == pytest.approx(20.0, abs=0.05)


def test_integrated_loudness_gates_silence_and_short_clips():
    assert serve_tts.integrated_loudness(np.zeros(RATE, dtype=np.float32), RATE) == -np.inf
    assert serve_tts.integrated_loudness(tone(seconds=0.3), RATE) == -np.inf  # shorter than one 400 ms block
    # about -83 LUFS: below the -70 LUFS absolute gate
    assert serve_tts.integrated_loudness(tone(seconds=2.0, freq=997.0, amplitude=1e-4), RATE) == -np.inf


def test_integrated_loudness_relative_gate_ignores_quiet_tail():
    speech = tone(seconds=2.0, freq=997.0, amplitude=0.5)
    # The tail (-66 LUFS) passes the absolute gate; averaged in, it would halve the power (-3 dB). Only the blocks
    # straddling the transition still count.
    with_tail = np.concatenate([speech, tone(seconds=2.0, freq=997.0, amplitude=0.0005)])
    assert serve_tts.integrated_loudness(with_tail, RATE) == pytest.approx(
        serve_tts.integrated_loudness(speech, RATE), abs=0.5
    )


def test_normalize_loudness_reaches_target():
    audio, loudness, limited = serve_tts.normalize_loudness(tone(seconds=2.0, amplitude=0.05), RATE, -20.0)
    assert not limited
    assert loudness == pytest.approx(-20.0, abs=1e-6)
    assert serve_tts.integrated_loudness(audio, RATE) == pytest.approx(-20.0, abs=0.01)
    assert audio.dtype == np.float32


def test_normalize_loudness_caps_peak():
    audio, loudness, limited = serve_tts.normalize_loudness(tone(seconds=2.0, amplitude=0.05), RATE, 0.0)
    assert limited
    assert np.abs(audio).max() == pytest.approx(serve_tts.PEAK_CEILING, rel=1e-4)
    assert loudness < 0.0


def test_normalize_loudness_leaves_silence_alone():
    silence = np.zeros(RATE, dtype=np.float32)
    audio, loudness, limited = serve_tts.normalize_loudness(silence, RATE, -20.0)
    assert audio is silence
    assert loudness == -np.inf
    assert not limited
