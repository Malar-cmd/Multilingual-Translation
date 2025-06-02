import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
from googletrans import Translator       
from gtts import gTTS
from IPython.display import Audio, display
import os
import io

# Load Whisper model
model = whisper.load_model("base")

# Initialize translator
translator = Translator()

def record_until_silence(filename="user_audio.wav", samplerate=16000, silence_threshold=0.02, silence_duration=1.0):
    print("🎙️ Start speaking... (recording will auto-stop after 1 second of silence)")

    buffer = []
    chunk_duration = 0.2  # seconds
    chunk_size = int(samplerate * chunk_duration)
    silence_chunks_required = int(silence_duration / chunk_duration)
    silent_chunks = 0

    def is_silent(audio_chunk):
        volume_norm = np.linalg.norm(audio_chunk)
        return volume_norm < silence_threshold

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', blocksize=chunk_size) as stream:
            while True:
                audio_chunk, _ = stream.read(chunk_size)
                buffer.append(audio_chunk)

                if is_silent(audio_chunk):
                    silent_chunks += 1
                else:
                    silent_chunks = 0  # reset on speech

                if silent_chunks >= silence_chunks_required:
                    break

        # Combine and save
        full_audio = np.concatenate(buffer, axis=0)
        if full_audio.shape[0] == 0:
            print("⚠️ No speech detected!")
            return None

        write(filename, samplerate, (full_audio * 32767).astype(np.int16))  # 16-bit PCM
        print(f"📂 Saved audio to: {filename}")
        print(f"📏 File size: {os.path.getsize(filename)} bytes" if os.path.exists(filename) else "❌ File not saved")

        if not os.path.exists(filename):
            print("❌ Error: Audio file was not saved!")
            return None

        print(f"✅ Recording complete. Saved as: {filename} ({os.path.getsize(filename)} bytes)")
        return filename

    except Exception as e:
        print(f"❌ Error during recording: {e}")
        return None


def transcribe_audio(audio_path):
    print(f"🔊 Transcribing: {audio_path}")
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError("Audio file does not exist.")

        result = model.transcribe(audio_path, language="en")
        return result["text"]
    except Exception as e:
        return f"Transcription Error: {e}"


def translate_text(text, target_lang_code):
    print(f"🌐 Translating to: {target_lang_code}")
    try:
        translated = translator.translate(text, dest=target_lang_code)
        return translated.text
    except Exception as e:
        return f"Translation Error: {e}"


def text_to_speech(text, lang_code):
    print(f"🎵 Converting to speech ({lang_code})...")
    try:
        tts = gTTS(text=text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return Audio(fp.read(), autoplay=True)
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None


def main():
    save_path = save_path = os.path.join(os.getcwd(), "user_audio.wav")  # Saves in your current folder

    audio_path = record_until_silence(filename=save_path)

    if not audio_path or not os.path.exists(audio_path):
        print("❌ Audio file not found. Please try again.")
        return

    english_text = transcribe_audio(audio_path)
    print("\n📝 Transcribed Text:\n", english_text)

    if "Transcription Error" in english_text:
        print("❌ Stopping due to transcription failure.")
        return

    target_lang_code = input("🌐 Enter target language code (e.g., 'hi' for Hindi): ").strip()

    translated = translate_text(english_text, target_lang_code)
    print("\n✅ Translated Text:\n", translated)

    # Convert translated text to speech
    tts_audio = text_to_speech(translated, target_lang_code)
    display(tts_audio)

if __name__ == "__main__":
    main()
