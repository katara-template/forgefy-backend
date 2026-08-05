#include "PcmSink.h"

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "util/Log.h"

namespace {
// Lets the sidecar validate it is talking to a matching bot build before it
// starts interpreting bytes as audio.
constexpr char kMagic[8] = {'F', 'G', 'F', 'Y', 'P', 'C', 'M', '1'};
} // namespace

PcmSink::~PcmSink() {
    stop();
}

bool PcmSink::start(const string& socketPath, int timeoutSecs) {
    if (socketPath.size() >= sizeof(sockaddr_un::sun_path)) {
        Log::error("audio socket path is too long: " + socketPath);
        return false;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

    // The sidecar binds the socket before spawning us, but container start
    // ordering is not guaranteed — retry rather than die on a cold race.
    for (int elapsed = 0; elapsed <= timeoutSecs; ++elapsed) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd == -1) {
            Log::error(string("socket() failed: ") + strerror(errno));
            return false;
        }

        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            m_fd = fd;
            m_connected.store(true);
            m_running.store(true);
            m_writer = thread(&PcmSink::writerLoop, this);
            Log::success("audio socket connected: " + socketPath);
            return true;
        }

        close(fd);
        if (elapsed < timeoutSecs) sleep(1);
    }

    Log::error("timed out connecting to audio socket " + socketPath);
    return false;
}

bool PcmSink::sendHeader(uint32_t sampleRate, uint32_t channels) {
    if (!m_connected.load()) return false;
    if (m_headerSent.exchange(true)) return true;  // already sent

    char header[16];
    memcpy(header, kMagic, sizeof(kMagic));
    // Little-endian on the wire; both ends are x86_64 but be explicit anyway.
    for (int i = 0; i < 4; ++i) {
        header[8 + i]  = static_cast<char>((sampleRate >> (8 * i)) & 0xFF);
        header[12 + i] = static_cast<char>((channels   >> (8 * i)) & 0xFF);
    }

    // Written inline rather than queued so it always precedes the first
    // audio frame regardless of writer-thread scheduling.
    lock_guard<mutex> lock(m_mutex);
    return writeAll(header, sizeof(header));
}

void PcmSink::push(const char* data, size_t len) {
    if (!m_running.load() || len == 0) return;

    {
        lock_guard<mutex> lock(m_mutex);

        while (m_queuedBytes + len > kMaxQueuedBytes && !m_queue.empty()) {
            m_droppedBytes.fetch_add(m_queue.front().size());
            m_queuedBytes -= m_queue.front().size();
            m_queue.pop_front();
        }

        m_queue.emplace_back(data, data + len);
        m_queuedBytes += len;
    }
    m_cv.notify_one();
}

void PcmSink::writerLoop() {
    while (true) {
        vector<char> chunk;
        {
            unique_lock<mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return !m_queue.empty() || !m_running.load(); });

            if (m_queue.empty()) {
                if (!m_running.load()) break;
                continue;
            }

            chunk = std::move(m_queue.front());
            m_queue.pop_front();
            m_queuedBytes -= chunk.size();
        }

        if (!writeAll(chunk.data(), chunk.size())) {
            // The sidecar is gone. Stop accepting audio; the SDK lifecycle
            // will wind the meeting down separately.
            m_connected.store(false);
            m_running.store(false);
            break;
        }
    }
}

bool PcmSink::writeAll(const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        // MSG_NOSIGNAL: a dead sidecar must surface as EPIPE, not SIGPIPE.
        ssize_t n = send(m_fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n > 0) {
            sent += static_cast<size_t>(n);
            continue;
        }
        if (n == -1 && errno == EINTR) continue;
        Log::error(string("audio socket write failed: ") + strerror(errno));
        return false;
    }
    return true;
}

void PcmSink::stop() {
    if (!m_running.exchange(false) && m_fd == -1) return;

    m_cv.notify_all();
    if (m_writer.joinable()) m_writer.join();

    if (m_fd != -1) {
        // Half-close so the sidecar sees clean EOF and can finalize the
        // Deepgram stream instead of treating it as a dropped connection.
        shutdown(m_fd, SHUT_WR);
        close(m_fd);
        m_fd = -1;
    }
    m_connected.store(false);

    uint64_t dropped = m_droppedBytes.load();
    if (dropped > 0)
        Log::error("dropped " + to_string(dropped) + " bytes of audio (sidecar fell behind)");
}
