#include "AudioDelegate.h"

#include "util/Log.h"

void AudioDelegate::onMixedAudioRawDataReceived(AudioRawData* data) {
    if (m_separateParticipants) return;
    forward(data);
}

void AudioDelegate::onOneWayAudioRawDataReceived(AudioRawData* data, uint32_t node_id) {
    if (!m_separateParticipants) return;

    // Per-participant capture is deliberately not finished: routing several
    // speakers into one PCM socket would interleave them unusably. Doing this
    // properly means one framed sub-stream (or one socket) per node_id, and a
    // Deepgram connection per speaker on the sidecar side. Until that exists,
    // FORGEFY_SEPARATE_PARTICIPANT_AUDIO only proves the callbacks arrive.
    static thread_local uint32_t lastNode = 0;
    if (node_id != lastNode) {
        lastNode = node_id;
        Log::info("one-way audio from node " + std::to_string(node_id));
    }
    forward(data);
}

void AudioDelegate::onShareAudioRawDataReceived(AudioRawData* data, unsigned int user_id) {
    // Audio from shared screen content (a played video, say). Excluded so the
    // transcript reflects what people said, not what a slide deck played.
}

void AudioDelegate::forward(AudioRawData* data) {
    if (!data || !m_sink) return;

    char* buffer = data->GetBuffer();
    unsigned int len = data->GetBufferLen();
    if (!buffer || len == 0) return;

    if (!m_formatAnnounced.exchange(true)) {
        unsigned int sampleRate = data->GetSampleRate();
        unsigned int channels   = data->GetChannelNum();

        // Never hardcoded on either side: the sidecar configures Deepgram's
        // encoding/sample_rate from what actually arrives here.
        m_sink->sendHeader(sampleRate, channels);
        Status::emitAudioFormat(static_cast<int>(sampleRate), static_cast<int>(channels));
        Log::success("raw audio flowing at " + std::to_string(sampleRate) + "Hz, " +
                     std::to_string(channels) + "ch");
    }

    m_sink->push(buffer, len);
}
