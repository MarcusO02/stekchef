# voice_dialog.py
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import re
import argparse
# Hotkey: prefers keyboard, falls back to ENTER if unavailable
def _wait_for_space():
    try:
        import keyboard
        print("\n[VOICE] Press SPACE to speak...")
        keyboard.wait("space")
    except Exception:
        input("\n[VOICE] Press ENTER to speak...")

# Google Cloud
from google.cloud import speech
from google.cloud import texttospeech as tts
from google.oauth2 import service_account

# ---------------- Config ----------------
# Leave KEY_PATH as None to use GOOGLE_APPLICATION_CREDENTIALS env var or local JSON if hardcoding is prefered.
KEY_PATH = None
# KEY_PATH = r"C:\Users\elias\Downloads\hubert-475815-8758c3819846.json"

LANGUAGE_CODE = "en-US"  
TTS_VOICE = "en-US-Neural2-C" if LANGUAGE_CODE == "en-US" else "sv-SE-Standard-A"

SAMPLE_RATE = 16000
CHANNELS = 1
INPUT_DEVICE = None         
MAX_RECORD_SECONDS = 6.0

PHRASE_HINTS = [
    "start cooking the steak", "start", "begin", "cook",
    "rare", "medium rare", "medium", "medium well", "well done"
]

# Doneness map and synonyms
DONENESS_MAP = {
    "rare": 2,
    "medium rare": 4,
    "medium": 6,
    "medium well": 8,
    "well done": 10,
}
DONENESS_SYNONYMS = {
    "rare": ["rare", "raw"],
    "medium rare": ["medium rare", "medium-rare"],
    "medium": ["medium"],
    "medium well": ["medium well", "medium-well"],
    "well done": ["well done", "well-done"],
}

# ---------------- Clients ----------------
def _get_creds():
    if KEY_PATH:
        return service_account.Credentials.from_service_account_file(KEY_PATH)
    return None

_CREDS = _get_creds()
_SPEECH = speech.SpeechClient(credentials=_CREDS) if _CREDS else speech.SpeechClient()
_TTS = tts.TextToSpeechClient(credentials=_CREDS) if _CREDS else tts.TextToSpeechClient()

# --- helpers ---
def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-zåäö\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _compile_doneness_patterns():
    items = []
    for canonical, variants in DONENESS_SYNONYMS.items():
        human = canonical
        for v in variants:
            v_n = _normalize_text(v)
            token = re.escape(v_n).replace(r"\ ", r"[ -]?")
            rx = re.compile(rf"\b{token}\b")
            items.append((len(v_n.replace(" ", "")), rx, canonical, human))
    items.sort(key=lambda t: (-t[0], t[2]))
    return items

_DONENESS_PATTERNS = _compile_doneness_patterns()

# ---------------- TTS ----------------
def _tts_say(text: str, voice_name: str = None, speaking_rate: float = 1.0):
    if voice_name is None:
        voice_name = TTS_VOICE
    synthesis_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(language_code=LANGUAGE_CODE, name=voice_name)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, speaking_rate=speaking_rate)
    resp = _TTS.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
    pcm = np.frombuffer(resp.audio_content, dtype=np.int16)
    sd.play(pcm, 24000)
    sd.wait()

# ---------------- STT ----------------
def _asr_listen_once(max_seconds: float = MAX_RECORD_SECONDS) -> str:
    print(f"[VOICE] Listening... (up to {max_seconds:.1f}s)")

    rec_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="command_and_search",
        max_alternatives=1,
        profanity_filter=False,
        speech_contexts=[speech.SpeechContext(phrases=PHRASE_HINTS)],
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=rec_config,
        interim_results=True,
        single_utterance=False,
    )

    audio_q: "queue.Queue[bytes|None]" = queue.Queue()
    stop_ev = threading.Event()

    def mic_stream():
        block_dur = 0.2
        frames_per_read = int(SAMPLE_RATE * block_dur)
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=frames_per_read,
                device=INPUT_DEVICE,
                dtype="int16",
                channels=CHANNELS,
            ) as stream:
                start = time.time()
                while not stop_ev.is_set() and (time.time() - start) < max_seconds:
                    data, _ = stream.read(frames_per_read)
                    audio_q.put(bytes(data))
        except Exception as e:
            print(f"[VOICE] Mic error: {e}")
        finally:
            audio_q.put(None)

    t = threading.Thread(target=mic_stream, daemon=True)
    t.start()

    def audio_requests():
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    try:
        responses = _SPEECH.streaming_recognize(streaming_config, audio_requests())
        final_text = ""
        for resp in responses:
            for result in resp.results:
                if result.is_final:
                    final_text = result.alternatives[0].transcript.strip()
                    stop_ev.set(); t.join(timeout=0.5)
                    return final_text
        stop_ev.set(); t.join(timeout=0.5)
        return final_text
    except TypeError:
        def requests_with_config_first():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            yield from audio_requests()
        try:
            responses = _SPEECH.streaming_recognize(requests=requests_with_config_first())
            final_text = ""
            for resp in responses:
                for result in resp.results:
                    if result.is_final:
                        final_text = result.alternatives[0].transcript.strip()
                        stop_ev.set(); t.join(timeout=0.5)
                        return final_text
            stop_ev.set(); t.join(timeout=0.5)
            return final_text
        except Exception as e2:
            stop_ev.set(); t.join(timeout=0.5)
            print(f"[VOICE] ASR error (fallback): {e2}")
            return ""
    except Exception as e:
        stop_ev.set(); t.join(timeout=0.5)
        print(f"[VOICE] ASR error: {e}")
        return ""

# ---------------- Intent parsing ----------------
def _parse_intent(text: str):
    s = _normalize_text(text)

    # start intent
    if re.search(r"\bstart\b", s) or re.search(r"\bbegin\b", s) or re.search(r"\bcook\b", s) and re.search(r"\bsteak\b", s):
        return ("start", None)

    # doneness: try longest patterns first
    for _, pat, canonical, human in _DONENESS_PATTERNS:
        if pat.search(s):
            seconds = DONENESS_MAP.get(human, DONENESS_MAP.get(canonical))
            return ("doneness", (canonical, human, seconds))

    if re.search(r"\b(stop|cancel|avbryt)\b", s):
        return ("cancel", None)

    return ("unknown", None)

def run_start_and_doneness_dialog(use_voice: bool = True) -> tuple[float, str]:
    # Wait for 'start'
    # If voice input is disabled, return default doneness "medium rare"
    if not use_voice:
        print("[VOICE] Voice input disabled — using default: medium rare")
        return float(DONENESS_MAP["medium rare"]), "medium rare"
    while True:
        _tts_say("Press space and say start when you are ready.")
        _wait_for_space()
        heard = _asr_listen_once()
        print(f"[VOICE] Heard: {heard!r}")
        intent, data = _parse_intent(heard)
        if intent == "start":
            break
        _tts_say("I did not hear start. Let's try again.")

    # Ask doneness
    while True:
        _tts_say("How would you like your steak cooked?")
        _wait_for_space()
        heard2 = _asr_listen_once()
        print(f"[VOICE] Heard: {heard2!r}")
        intent2, data2 = _parse_intent(heard2)
        if intent2 == "doneness" and data2 and data2[2] is not None:
            canonical, human, seconds = data2
            # Reject 'well done' explicitly
            if canonical == "well done":
                _tts_say("Are you a child? Who eats well done steak? Lets try again or get out of my fucking kitchen.")
                continue
            _tts_say(f"You said {human}. Is that correct?")
            _wait_for_space()
            confirm = _asr_listen_once().lower()
            if "yes" in confirm:
                _tts_say(f"Great, watch me make this masterpiece.")
                return float(seconds), human
            else:
                _tts_say("Okay, let's try again.")
        else:
            _tts_say("Sorry, please say rare, medium rare, medium, medium well, or well done.")
