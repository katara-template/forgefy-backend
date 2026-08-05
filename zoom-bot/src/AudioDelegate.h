#ifndef FORGEFY_ZOOM_BOT_AUDIODELEGATE_H
#define FORGEFY_ZOOM_BOT_AUDIODELEGATE_H

#include <atomic>

#include "zoom_sdk_raw_data_def.h"
#include "rawdata/rawdata_audio_helper_interface.h"

#include "PcmSink.h"

using namespace ZOOMSDK;

/**
 * Receives raw PCM from the Meeting SDK and forwards it to the sidecar.
 *
 * MVP captures the *mixed* stream — every participant summed into one channel,
 * which is what Deepgram needs for a single transcript. Zoom also offers
 * per-participant streams via onOneWayAudioRawDataReceived (one callback per
 * speaker, keyed by node_id); wiring those up would let us attribute each
 * utterance to a speaker without relying on diarization, at the cost of one
 * Deepgram connection per participant. See the note in that method body.
 */
class AudioDelegate : public IZoomSDKAudioRawDataDelegate {
public:
    explicit AudioDelegate(PcmSink* sink, bool separateParticipants)
        : m_sink(sink), m_separateParticipants(separateParticipants) {}

    void onMixedAudioRawDataReceived(AudioRawData* data) override;
    void onOneWayAudioRawDataReceived(AudioRawData* data, uint32_t node_id) override;
    void onShareAudioRawDataReceived(AudioRawData* data, unsigned int user_id) override;

    // Interpreter channels are a paid Zoom feature we do not consume; the
    // override exists only to satisfy the interface.
    void onOneWayInterpreterAudioRawDataReceived(AudioRawData* data,
                                                 const zchar_t* pLanguageName) override {}

private:
    /** Emit the audio format on the first frame, then stream. */
    void forward(AudioRawData* data);

    PcmSink* m_sink;
    bool m_separateParticipants;
    std::atomic<bool> m_formatAnnounced{false};
};

#endif //FORGEFY_ZOOM_BOT_AUDIODELEGATE_H
