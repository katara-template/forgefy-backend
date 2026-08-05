#ifndef FORGEFY_ZOOM_BOT_PCMSINK_H
#define FORGEFY_ZOOM_BOT_PCMSINK_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace std;

/**
 * Ships raw PCM from the SDK's audio callback to the Python sidecar over a
 * Unix domain socket.
 *
 * Two design points that matter in a real meeting:
 *
 * 1. We are the *client*. The sidecar binds and listens before spawning this
 *    process, so there is never a window where audio arrives with nowhere to
 *    go. (Zoom's own sample inverts this and writes to an unconnected fd.)
 *
 * 2. The SDK delivers audio on its own callback thread and expects that
 *    callback to return promptly. A blocking write() to a slow reader would
 *    stall the SDK and drop audio at the source. So push() only enqueues, and
 *    a dedicated writer thread drains the queue. If the sidecar falls
 *    permanently behind, we drop the oldest audio rather than grow without
 *    bound — a bounded gap in the transcript beats an OOM kill mid-meeting.
 */
class PcmSink {
public:
    PcmSink() = default;
    ~PcmSink();

    /**
     * Connect to socketPath and start the writer thread.
     * Retries for up to timeoutSecs to tolerate sidecar startup ordering.
     */
    bool start(const string& socketPath, int timeoutSecs = 30);

    /** Send the stream header once the SDK reports its actual audio format. */
    bool sendHeader(uint32_t sampleRate, uint32_t channels);

    /** Enqueue a buffer. Never blocks. Safe to call from the SDK thread. */
    void push(const char* data, size_t len);

    /** Flush what is queued, then close. Idempotent. */
    void stop();

    bool connected() const { return m_connected.load(); }
    uint64_t droppedBytes() const { return m_droppedBytes.load(); }

private:
    void writerLoop();
    bool writeAll(const char* data, size_t len);

    int m_fd = -1;

    thread m_writer;
    mutex m_mutex;
    condition_variable m_cv;
    deque<vector<char>> m_queue;
    size_t m_queuedBytes = 0;

    atomic<bool> m_connected{false};
    atomic<bool> m_running{false};
    atomic<bool> m_headerSent{false};
    atomic<uint64_t> m_droppedBytes{0};

    // ~16 s of 32 kHz mono PCM-16. Large enough to ride out a Deepgram
    // reconnect, small enough to bound container memory.
    static constexpr size_t kMaxQueuedBytes = 1024 * 1024;
};

#endif //FORGEFY_ZOOM_BOT_PCMSINK_H
