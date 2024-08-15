import fakeyou
import pyaudio
import time
import wave
import io
from threading import Thread


def read(fy, content, voice=None):
    if voice is None:
        voice = "weight_k7seyhbcj7877kderbsrkj9nt"

    token = fy.make_tts_job(content, voice)
    wav = fy.tts_poll(token)
    while wav.status != "complete_success":
        time.sleep(1)
        wav = fy.tts_poll(token)
    audio_content = wav.content

    with wave.open(io.BytesIO(audio_content), "rb") as f:
        width = f.getsampwidth()
        channels = f.getnchannels()
        rate = f.getframerate()
    pa = pyaudio.PyAudio()
    pa_stream = pa.open(
        format=pyaudio.get_format_from_width(width),
        channels=channels,
        rate=rate,
        output=True,
    )
    pa_stream.write(audio_content)


if __name__ == "__main__":
    # fy = fakeyou.FakeYou()
    # read(fy, "Hello there, what's up?")

    from RealtimeTTS import TextToAudioStream, CoquiEngine

    engine = CoquiEngine()
    stream = TextToAudioStream(engine)

    stream.feed("Hello world, How are you today?")
    stream.play_async()
