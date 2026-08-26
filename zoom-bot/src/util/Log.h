#ifndef FORGEFY_ZOOM_BOT_LOG_H
#define FORGEFY_ZOOM_BOT_LOG_H

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

/**
 * Two output streams with different consumers:
 *
 *   stderr — human-readable diagnostics, for `docker logs`
 *   stdout — one JSON object per line, consumed by the Python sidecar and
 *            translated into session state transitions
 *
 * Keeping them separate means noisy SDK logging can never corrupt the status
 * channel the orchestrator depends on.
 */
class Log {
public:
    static void info(const string& message)    { cerr << "[info]  " << message << endl; }
    static void success(const string& message) { cerr << "[ok]    " << message << endl; }
    static void error(const string& message)   { cerr << "[error] " << message << endl; }
};

class Status {
public:
    /** Emit a lifecycle transition, e.g. Status::emit("in_meeting"). */
    static void emit(const string& status, const string& detail = "") {
        ostringstream json;
        json << R"({"event":"status","status":")" << escape(status) << '"';
        if (!detail.empty())
            json << R"(,"detail":")" << escape(detail) << '"';
        json << R"(,"ts":)" << epochMillis() << '}';
        writeLine(json.str());
    }

    /** Emit the negotiated audio format once raw capture begins. */
    static void emitAudioFormat(int sampleRate, int channels) {
        ostringstream json;
        json << R"({"event":"audio_format","sample_rate":)" << sampleRate
             << R"(,"channels":)" << channels
             << R"(,"ts":)" << epochMillis() << '}';
        writeLine(json.str());
    }

private:
    static void writeLine(const string& line) {
        // Unbuffered-by-line: the sidecar reads this stream continuously and a
        // stalled status event would delay a session transition.
        cout << line << endl;
        cout.flush();
    }

    static long long epochMillis() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    static string escape(const string& in) {
        string out;
        out.reserve(in.size());
        for (char c : in) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:   out += c;      break;
            }
        }
        return out;
    }
};

#endif //FORGEFY_ZOOM_BOT_LOG_H
