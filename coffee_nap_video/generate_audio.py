#!/usr/bin/env python3
"""
Generates 16 synchronized PT-BR neural voiceover segments (one per 5.0s scene),
extracts exact word-level timestamps for kinetic subtitles, synthesizes a
cinematic ambient soundtrack + 5-second transition SFX, and builds the final
mastered 80.0s audio track + timeline.json.
"""

import asyncio
import json
import math
import os
import subprocess
import wave
from pathlib import Path

import edge_tts
import numpy as np
from scipy.signal import butter, filtfilt

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
BUILD_DIR = BASE_DIR / "build"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BUILD_DIR.mkdir(parents=True, exist_ok=True)

VOICE = "pt-BR-AntonioNeural"
SCENE_DURATION = 5.0  # Exact 5.0s per scene
SAMPLE_RATE = 48000

SCENES = [
    {
        "id": 1,
        "image": "scene_01.jpg",
        "text": "Se você já tirou um cochilo de quarenta e cinco minutos à tarde e acordou destruído...",
        "display_text": "Se você já tirou um cochilo de 45 minutos à tarde e acordou destruído...",
        "rate": "+8%",
        "badge": "45 MIN DE COCHILO",
        "badge_color": "#EF4444",
        "hud_type": "badge",
        "motion": "zoom_in_center",
        "highlights": ["45", "minutos", "destruído"],
    },
    {
        "id": 2,
        "image": "scene_02.jpg",
        "text": "o problema não é falta de força de vontade — é bioquímica mal sincronizada.",
        "display_text": "o problema não é falta de força de vontade — é bioquímica mal sincronizada.",
        "rate": "+8%",
        "badge": "SINCRONIA BIOQUÍMICA",
        "badge_color": "#F59E0B",
        "hud_type": "badge",
        "motion": "zoom_out_center",
        "highlights": ["bioquímica", "mal", "sincronizada"],
    },
    {
        "id": 3,
        "image": "scene_03.jpg",
        "text": "Um estudo clássico da NASA testou cochilos programados em pilotos de longo curso.",
        "display_text": "Um estudo clássico da NASA testou cochilos programados em pilotos de longo curso.",
        "rate": "+8%",
        "badge": "ESTUDO NASA • 1995",
        "badge_color": "#06B6D4",
        "hud_type": "badge",
        "motion": "pan_right_zoom",
        "highlights": ["NASA", "cochilos", "pilotos"],
    },
    {
        "id": 4,
        "image": "scene_04.jpg",
        "text": "Vinte e seis minutos elevaram o desempenho em trinta e quatro por cento e o alerta em cinquenta e quatro.",
        "display_text": "26 minutos elevaram o desempenho em +34% e o alerta em +54%.",
        "rate": "+12%",
        "badge": "DADOS NASA • 26 MIN",
        "badge_color": "#10B981",
        "hud_type": "nasa_bars",
        "motion": "zoom_in_center",
        "highlights": ["vinte", "seis", "desempenho", "trinta", "quatro", "alerta", "cinquenta"],
    },
    {
        "id": 5,
        "image": "scene_05.jpg",
        "text": "Mas existe um limite crítico: passou de trinta minutos, o efeito se inverte totalmente.",
        "display_text": "Mas existe um limite crítico: passou de 30 minutos, o efeito se inverte totalmente.",
        "rate": "+10%",
        "badge": "PONTO DE VIRADA • 30 MIN",
        "badge_color": "#EF4444",
        "hud_type": "cliff_gauge",
        "motion": "pan_up_zoom",
        "highlights": ["limite", "crítico", "trinta", "inverte"],
    },
    {
        "id": 6,
        "image": "scene_06.jpg",
        "text": "A explicação acontece dentro dos seus neurônios desde o segundo em que você acorda.",
        "display_text": "A explicação acontece dentro dos seus neurônios desde o segundo em que você acorda.",
        "rate": "+10%",
        "badge": "NEUROCIÊNCIA DO SONO",
        "badge_color": "#8B5CF6",
        "hud_type": "badge",
        "motion": "zoom_in_center",
        "highlights": ["neurônios", "acorda"],
    },
    {
        "id": 7,
        "image": "scene_07.jpg",
        "text": "A cada hora acordado, seu cérebro acumula adenosina: a molécula química da fadiga.",
        "display_text": "A cada hora acordado, seu cérebro acumula adenosina: a molécula química da fadiga.",
        "rate": "+10%",
        "badge": "ADENOSINA ↑ ACÚMULO",
        "badge_color": "#F97316",
        "hud_type": "pressure_meter",
        "motion": "pan_down_zoom",
        "highlights": ["adenosina", "molécula", "fadiga"],
    },
    {
        "id": 8,
        "image": "scene_08.jpg",
        "text": "Ela se encaixa nos receptores cerebrais e cria aquela pressão pesada de sono à tarde.",
        "display_text": "Ela se encaixa nos receptores cerebrais e cria aquela pressão pesada de sono à tarde.",
        "rate": "+10%",
        "badge": "RECEPTORES SATURADOS",
        "badge_color": "#EF4444",
        "hud_type": "badge",
        "motion": "zoom_in_tight",
        "highlights": ["receptores", "pressão", "sono"],
    },
    {
        "id": 9,
        "image": "scene_09.jpg",
        "text": "Passando de trinta minutos, você mergulha no sono profundo e ativa a inércia do sono.",
        "display_text": "Passando de 30 minutos, você mergulha no sono profundo e ativa a inércia do sono.",
        "rate": "+10%",
        "badge": "SONO PROFUNDO • INÉRCIA",
        "badge_color": "#3B82F6",
        "hud_type": "badge",
        "motion": "pan_down_zoom",
        "highlights": ["trinta", "sono", "profundo", "inércia"],
    },
    {
        "id": 10,
        "image": "scene_10.jpg",
        "text": "Acordar no meio dessa fase trava seu raciocínio numa névoa mental por quase uma hora.",
        "display_text": "Acordar no meio dessa fase trava seu raciocínio numa névoa mental por quase 1 hora.",
        "rate": "+10%",
        "badge": "NÉVOA MENTAL • ATÉ 60 MIN",
        "badge_color": "#94A3B8",
        "hud_type": "badge",
        "motion": "zoom_out_center",
        "highlights": ["névoa", "mental", "hora"],
    },
    {
        "id": 11,
        "image": "scene_11.jpg",
        "text": "Já um cochilo curto de vinte minutos limpa parte da adenosina antes de você afundar.",
        "display_text": "Já um cochilo curto de 20 minutos limpa parte da adenosina antes de você afundar.",
        "rate": "+10%",
        "badge": "ZONA IDEAL • 20 MIN",
        "badge_color": "#10B981",
        "hud_type": "badge",
        "motion": "pan_up_zoom",
        "highlights": ["vinte", "minutos", "limpa", "adenosina"],
    },
    {
        "id": 12,
        "image": "scene_12.jpg",
        "text": "E o truque validado em laboratório é o Coffee Nap: tome um café forte antes de deitar.",
        "display_text": "E o truque validado em laboratório é o Coffee Nap: tome um café forte antes de deitar.",
        "rate": "+10%",
        "badge": "PROTOCOLO: COFFEE NAP",
        "badge_color": "#F59E0B",
        "hud_type": "badge",
        "motion": "pan_down_zoom",
        "highlights": ["Coffee", "Nap", "café", "forte"],
    },
    {
        "id": 13,
        "image": "scene_13.jpg",
        "text": "Ajuste o alarme para vinte minutos exatos e feche os olhos, mesmo sem apagar por completo.",
        "display_text": "Ajuste o alarme para 20 minutos exatos e feche os olhos, mesmo sem apagar por completo.",
        "rate": "+10%",
        "badge": "CRONÔMETRO EXATO",
        "badge_color": "#14B8A6",
        "hud_type": "timer_20",
        "motion": "zoom_out_center",
        "highlights": ["alarme", "vinte", "minutos", "exatos"],
    },
    {
        "id": 14,
        "image": "scene_14.jpg",
        "text": "A cafeína leva cerca de vinte minutos para percorrer o sangue e chegar ao seu cérebro.",
        "display_text": "A cafeína leva cerca de 20 minutos para percorrer o sangue e chegar ao seu cérebro.",
        "rate": "+10%",
        "badge": "ABSORÇÃO • ~20 MIN",
        "badge_color": "#F59E0B",
        "hud_type": "badge",
        "motion": "pan_up_zoom",
        "highlights": ["cafeína", "vinte", "minutos", "cérebro"],
    },
    {
        "id": 15,
        "image": "scene_15.jpg",
        "text": "Quando o alarme toca, o sono esvaziou os receptores e a cafeína bloqueia todos eles juntos.",
        "display_text": "Quando o alarme toca, o sono esvaziou os receptores e a cafeína bloqueia todos eles juntos.",
        "rate": "+10%",
        "badge": "SINCRONIA PERFEITA",
        "badge_color": "#10B981",
        "hud_type": "badge",
        "motion": "zoom_in_tight",
        "highlights": ["esvaziou", "receptores", "cafeína", "bloqueia"],
    },
    {
        "id": 16,
        "image": "scene_16.jpg",
        "text": "Energia dobrada sem rebote, faça antes das três, salve e teste amanhã!",
        "display_text": "Energia dobrada sem rebote, faça antes das 15h, salve e teste amanhã!",
        "rate": "+12%",
        "badge": "CAFÉ → 20 MIN → PICO DE FOCO",
        "badge_color": "#F59E0B",
        "hud_type": "closing_card",
        "motion": "zoom_out_center",
        "highlights": ["Energia", "dobrada", "três", "Salva", "amanhã"],
    },
]


async def synth_scene(scene: dict) -> dict:
    sid = scene["id"]
    mp3_path = AUDIO_DIR / f"scene_{sid:02d}.mp3"
    wav_path = AUDIO_DIR / f"scene_{sid:02d}.wav"

    base_rate_val = int(scene["rate"].replace("%", "").replace("+", ""))
    words = []
    speech_end = 99.0

    for attempt in range(4):
        rate_str = f"+{base_rate_val + attempt * 4}%"
        communicate = edge_tts.Communicate(
            scene["text"],
            VOICE,
            rate=rate_str,
            boundary="WordBoundary",
        )
        words = []
        with open(mp3_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start_s = chunk["offset"] / 10_000_000.0
                    dur_s = chunk["duration"] / 10_000_000.0
                    words.append({
                        "word": chunk["text"],
                        "start": start_s,
                        "end": start_s + dur_s,
                    })

        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(mp3_path),
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                str(wav_path),
            ],
            check=True,
        )
        speech_end = (words[-1]["end"] + 0.14) if words else 4.5

        if speech_end <= 4.72:
            print(f"Scene {sid:02d}: speech_end={speech_end:.2f}s (rate {rate_str}, {len(words)} words)")
            break
        else:
            print(f"Scene {sid:02d}: speech_end={speech_end:.2f}s too long at {rate_str}, retrying slightly faster...")

    return {
        "id": sid,
        "duration": speech_end,
        "wav_path": str(wav_path),
        "words": words,
    }


def read_wav_float(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return data


def lowpass(signal: np.ndarray, cutoff: float, sr: int = SAMPLE_RATE, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff / (sr * 0.5), btype="low")
    return filtfilt(b, a, signal).astype(np.float32)


def highpass(signal: np.ndarray, cutoff: float, sr: int = SAMPLE_RATE, order: int = 2) -> np.ndarray:
    b, a = butter(order, cutoff / (sr * 0.5), btype="high")
    return filtfilt(b, a, signal).astype(np.float32)


def synth_soundtrack(total_duration: float) -> np.ndarray:
    """
    Synthesizes a warm, modern scientific ambient electronic soundtrack (stereo)
    with harmonic progression that shifts from moody/curious (0-50s) to bright/uplifting (50-80s),
    plus crisp subtle geometric pulse & 5-second scene transition whoosh/chime accents.
    """
    n_samples = int(total_duration * SAMPLE_RATE)
    t = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE

    # Chord progression: changes every 5.0s (aligned with visual scenes!)
    # Scenes 1-10 (0-50s): D minor scientific curiosity (Dm9 -> Bbmaj7 -> Fmaj7 -> Csus2 -> Gm7...)
    # Scenes 11-16 (50-80s): Uplifting resolve into bright F/C/G/A energy!
    chords_hz = [
        [146.83, 220.00, 261.63, 329.63],  # 01 Dm9
        [116.54, 174.61, 220.00, 293.66],  # 02 Bbmaj7
        [130.81, 196.00, 246.94, 293.66],  # 03 Cmaj9
        [174.61, 220.00, 261.63, 329.63],  # 04 Fmaj7
        [146.83, 207.65, 261.63, 329.63],  # 05 Dm/tension
        [146.83, 220.00, 261.63, 349.23],  # 06 Dm7
        [116.54, 174.61, 233.08, 293.66],  # 07 Bb
        [110.00, 164.81, 220.00, 261.63],  # 08 Am7
        [98.00,  146.83, 196.00, 233.08],  # 09 Gm7 (deep abyss)
        [110.00, 146.83, 196.00, 246.94],  # 10 Asus4 (fog)
        [174.61, 220.00, 261.63, 329.63],  # 11 Fmaj7 (cleanup bright!)
        [196.00, 246.94, 293.66, 392.00],  # 12 Gmaj (coffee!)
        [130.81, 196.00, 261.63, 329.63],  # 13 Cmaj (calm timer)
        [146.83, 220.00, 293.66, 369.99],  # 14 Dmaj (caffeine rising)
        [196.00, 246.94, 329.63, 392.00],  # 15 G6 (lock-in)
        [220.00, 277.18, 329.63, 440.00],  # 16 Amaj (triumphant close)
    ]

    pad_l = np.zeros(n_samples, dtype=np.float32)
    pad_r = np.zeros(n_samples, dtype=np.float32)

    for idx, freqs in enumerate(chords_hz):
        s0 = int(idx * SCENE_DURATION * SAMPLE_RATE)
        s1 = min(n_samples, int((idx + 1) * SCENE_DURATION * SAMPLE_RATE))
        seg_n = s1 - s0
        seg_t = np.arange(seg_n, dtype=np.float32) / SAMPLE_RATE

        # Smooth crossfade envelope
        env = np.ones(seg_n, dtype=np.float32)
        fade = int(0.6 * SAMPLE_RATE)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)

        seg_l = np.zeros(seg_n, dtype=np.float32)
        seg_r = np.zeros(seg_n, dtype=np.float32)
        for fi, f in enumerate(freqs):
            # Warm detuned sine + soft triangle harmonic
            detune = 0.003 * (fi - 1.5)
            osc_l = (
                0.7 * np.sin(2 * np.pi * f * (1 + detune) * seg_t)
                + 0.2 * np.sin(2 * np.pi * (f * 2.0) * (1 - detune) * seg_t)
                + 0.1 * np.sin(2 * np.pi * (f * 0.5) * seg_t)
            )
            osc_r = (
                0.7 * np.sin(2 * np.pi * f * (1 - detune) * seg_t)
                + 0.2 * np.sin(2 * np.pi * (f * 2.0) * (1 + detune) * seg_t)
                + 0.1 * np.sin(2 * np.pi * (f * 0.5) * seg_t)
            )
            seg_l += osc_l
            seg_r += osc_r

        # Gentle sub-bass pulse at 120 BPM (0.5s period)
        pulse = 0.55 + 0.45 * np.sin(2 * np.pi * 1.0 * seg_t) ** 2
        bass_f = freqs[0] * 0.5
        bass = 0.45 * np.sin(2 * np.pi * bass_f * seg_t) * pulse

        pad_l[s0:s1] += (seg_l * 0.18 + bass * 0.22) * env
        pad_r[s0:s1] += (seg_r * 0.18 + bass * 0.22) * env

        # Subtle plucky geometric arpeggio ticks every 0.5s (gives forward momentum)
        for beat in range(10):
            bt = beat * 0.5
            bs = s0 + int(bt * SAMPLE_RATE)
            plk_n = int(0.28 * SAMPLE_RATE)
            if bs + plk_n < n_samples:
                pt = np.arange(plk_n, dtype=np.float32) / SAMPLE_RATE
                note_f = freqs[beat % len(freqs)] * 2.0
                plk_env = np.exp(-pt * 14.0) * (1.0 - np.exp(-pt * 180.0))
                plk = np.sin(2 * np.pi * note_f * pt) * plk_env * 0.055
                pan = 0.35 + 0.3 * (beat % 2)
                pad_l[bs:bs + plk_n] += plk * (1.0 - pan)
                pad_r[bs:bs + plk_n] += plk * pan

    # Smooth filter for lush cinema sound
    pad_l = lowpass(pad_l, 1800.0)
    pad_r = lowpass(pad_r, 1800.0)

    # Subtle cinematic soft-whoosh + crystalline tick on every 5.0s scene cut (from scene 2..16)
    rng = np.random.default_rng(42)
    sfx_l = np.zeros(n_samples, dtype=np.float32)
    sfx_r = np.zeros(n_samples, dtype=np.float32)

    for idx in range(1, len(chords_hz)):
        cut_sample = int(idx * SCENE_DURATION * SAMPLE_RATE)
        w_dur = 0.36
        w_n = int(w_dur * SAMPLE_RATE)
        s0 = max(0, cut_sample - int(0.18 * SAMPLE_RATE))
        s1 = min(n_samples, s0 + w_n)
        actual_n = s1 - s0
        wt = np.linspace(0, 1, actual_n, dtype=np.float32)
        noise = rng.normal(0, 1, actual_n).astype(np.float32)
        noise = highpass(lowpass(noise, 1400.0), 220.0)
        w_env = np.sin(np.pi * wt) ** 2 * 0.09
        # Gentle crystal ping right at transition
        ping_f = chords_hz[idx][2] * 2.0
        ping_t = np.maximum(0.0, np.arange(actual_n, dtype=np.float32) / SAMPLE_RATE - 0.16)
        ping_mask = (np.arange(actual_n) / SAMPLE_RATE >= 0.16).astype(np.float32)
        ping = np.sin(2 * np.pi * ping_f * ping_t) * np.exp(-ping_t * 18.0) * ping_mask * 0.06
        sfx_l[s0:s1] += noise * w_env + ping * 0.9
        sfx_r[s0:s1] += noise * w_env * 0.9 + ping

    # Global fade in/out
    fade_in = int(0.5 * SAMPLE_RATE)
    fade_out = int(1.8 * SAMPLE_RATE)
    pad_l[:fade_in] *= np.linspace(0, 1, fade_in)
    pad_r[:fade_in] *= np.linspace(0, 1, fade_in)
    pad_l[-fade_out:] *= np.linspace(1, 0, fade_out)
    pad_r[-fade_out:] *= np.linspace(1, 0, fade_out)

    music = np.stack([pad_l + sfx_l, pad_r + sfx_r], axis=1)
    return music


async def main():
    results = []
    for sc in SCENES:
        res = await synth_scene(sc)
        results.append(res)

    total_duration = len(SCENES) * SCENE_DURATION  # 80.0s exact
    n_samples = int(total_duration * SAMPLE_RATE)
    voice_track = np.zeros(n_samples, dtype=np.float32)

    timeline = {
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "scene_duration": SCENE_DURATION,
        "total_duration": total_duration,
        "scenes": [],
    }

    for idx, (sc, res) in enumerate(zip(SCENES, results)):
        scene_start = idx * SCENE_DURATION
        audio_data = read_wav_float(res["wav_path"])

        # Center speech slightly inside the 5.0s slot with clean 0.12s lead-in
        lead_in = 0.12 if idx > 0 else 0.08
        max_len = int((SCENE_DURATION - lead_in - 0.08) * SAMPLE_RATE)
        if len(audio_data) > max_len:
            audio_data = audio_data[:max_len]

        start_sample = int((scene_start + lead_in) * SAMPLE_RATE)
        end_sample = min(n_samples, start_sample + len(audio_data))
        voice_track[start_sample:end_sample] += audio_data[: end_sample - start_sample]

        # Map word boundaries to global timeline
        # Also replace spelled-out numbers with crisp display tokens when matching caption chunks
        global_words = []
        for w in res["words"]:
            global_words.append({
                "word": w["word"],
                "start": round(scene_start + lead_in + w["start"], 3),
                "end": round(scene_start + lead_in + w["end"], 3),
            })

        timeline["scenes"].append({
            "id": sc["id"],
            "start": scene_start,
            "end": scene_start + SCENE_DURATION,
            "image": sc["image"],
            "text": sc["text"],
            "display_text": sc["display_text"],
            "badge": sc["badge"],
            "badge_color": sc["badge_color"],
            "hud_type": sc["hud_type"],
            "motion": sc["motion"],
            "highlights": sc["highlights"],
            "words": global_words,
        })

    # Normalize voice cleanly and apply subtle presence
    peak = np.max(np.abs(voice_track)) + 1e-8
    voice_track = (voice_track / peak) * 0.88

    # Synthesize stereo music bed + SFX
    music = synth_soundtrack(total_duration)

    # Gentle sidechain ducking envelope where voice is present
    abs_v = np.abs(voice_track)
    win = int(0.18 * SAMPLE_RATE)
    kernel = np.ones(win, dtype=np.float32) / win
    env_v = np.convolve(abs_v, kernel, mode="same")
    duck = np.clip(1.0 - env_v * 3.5, 0.42, 1.0)

    mix_l = voice_track + music[:, 0] * 0.34 * duck
    mix_r = voice_track + music[:, 1] * 0.34 * duck
    mix = np.stack([mix_l, mix_r], axis=1)

    # Soft limiter to prevent any clipping
    mix = np.tanh(mix * 1.05) * 0.94

    master_wav = AUDIO_DIR / "master_audio.wav"
    pcm = (np.clip(mix, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(master_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    timeline_path = BUILD_DIR / "timeline.json"
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    print(f"\nMaster audio saved to {master_wav} ({total_duration:.1f}s)")
    print(f"Timeline saved to {timeline_path}")


if __name__ == "__main__":
    asyncio.run(main())
