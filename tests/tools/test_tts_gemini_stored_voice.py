"""Stored Gemini voice IDs belong to Interactions, not prebuiltVoiceConfig."""
import base64
import io
import wave
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(b'\x00\x01' * 2400)
    return out.getvalue()


@pytest.mark.parametrize('response_shape', ['steps', 'output_audio'])
def test_stored_voice_uses_interactions_preserves_wav(tmp_path, wav_bytes, response_shape):
    from tools.tts_tool_providers import _generate_gemini_tts
    audio = {'type': 'audio', 'mime_type': 'audio/wav', 'data': base64.b64encode(wav_bytes).decode()}
    data = {'steps': [{'type': 'model_output', 'content': [audio]}]} if response_shape == 'steps' else {'output_audio': audio}
    response = MagicMock(status_code=200)
    response.json.return_value = data
    out = str(tmp_path / 'test.wav')
    with patch('tools.tts_tool._resolve_provider_key', return_value='test-key'), patch('requests.post', return_value=response) as post:
        assert _generate_gemini_tts('Hello', out, {'gemini': {'model': 'gemini-3.8-flash-lite-tts', 'voice': 'voice_test'}}) == out
    args, kw = post.call_args
    assert args[0].endswith('/interactions')
    assert kw['json']['generation_config']['speech_config'] == [{'voice': 'voice_test'}]
    assert kw['json']['response_format'] == {'type': 'audio'}
    assert (tmp_path / 'test.wav').read_bytes() == wav_bytes


def test_stored_voice_empty_audio_raises(tmp_path):
    from tools.tts_tool_providers import _generate_gemini_tts
    response = MagicMock(status_code=200)
    response.json.return_value = {'steps': []}
    with patch('tools.tts_tool._resolve_provider_key', return_value='test-key'), patch('requests.post', return_value=response):
        with pytest.raises(RuntimeError, match='audio'):
            _generate_gemini_tts('Hello', str(tmp_path / 'test.wav'), {'gemini': {'voice': 'voice_test'}})


def test_stored_voice_streamer_yields_pcm_not_wav(tmp_path, wav_bytes):
    from tools.tts_streaming import GeminiStreamer
    response = MagicMock(status_code=200)
    response.json.return_value = {'output_audio': {'mime_type': 'audio/wav', 'data': base64.b64encode(wav_bytes).decode()}}
    with patch('tools.tts_tool._resolve_provider_key', return_value='test-key'), patch('requests.post', return_value=response) as post:
        section = {'voice': 'voice_test', 'model': 'gemini-3.8-flash-lite-tts'}
        streamer = GeminiStreamer({'gemini': section}, section)
        audio = b''.join(streamer.stream('Hello'))
    assert post.call_args[0][0].endswith('/interactions')
    assert audio == b'\x00\x01' * 2400
    assert streamer.sample_rate == 24000
