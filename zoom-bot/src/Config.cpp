#include "Config.h"

#include <algorithm>
#include <cstdlib>
#include <cctype>

#include "util/Log.h"

namespace {

string envOr(const char* name, const string& fallback = "") {
    const char* raw = getenv(name);
    if (!raw) return fallback;
    string value(raw);
    if (value.empty()) return fallback;
    return value;
}

bool envBool(const char* name, bool fallback) {
    const char* raw = getenv(name);
    if (!raw || !*raw) return fallback;

    string value(raw);
    transform(value.begin(), value.end(), value.begin(),
              [](unsigned char c) { return tolower(c); });

    if (value == "1" || value == "true" || value == "yes" || value == "on")
        return true;
    if (value == "0" || value == "false" || value == "no" || value == "off")
        return false;

    Log::error(string(name) + " is not a boolean, using default");
    return fallback;
}

int envInt(const char* name, int fallback) {
    const char* raw = getenv(name);
    if (!raw || !*raw) return fallback;
    try {
        return stoi(string(raw));
    } catch (...) {
        Log::error(string(name) + " is not an integer, using default");
        return fallback;
    }
}

/** Strip everything but digits — tolerates "123 456 7890" and "123-456-7890". */
string digitsOnly(const string& in) {
    string out;
    for (char c : in)
        if (isdigit(static_cast<unsigned char>(c))) out += c;
    return out;
}

} // namespace

bool Config::load() {
    m_jwt           = envOr("ZOOM_SDK_JWT");
    m_meetingNumber = digitsOnly(envOr("ZOOM_MEETING_NUMBER"));
    m_password      = envOr("ZOOM_MEETING_PASSWORD");
    m_displayName   = envOr("ZOOM_DISPLAY_NAME", m_displayName);
    m_joinToken     = envOr("ZOOM_JOIN_TOKEN");
    m_zak           = envOr("ZOOM_ZAK");
    m_onBehalfToken = envOr("ZOOM_ON_BEHALF_TOKEN");
    m_zoomHost      = envOr("ZOOM_HOST", m_zoomHost);
    m_socketPath    = envOr("FORGEFY_AUDIO_SOCKET", m_socketPath);

    m_requireHostConsent    = envBool("FORGEFY_REQUIRE_HOST_CONSENT", true);
    m_separateParticipants  = envBool("FORGEFY_SEPARATE_PARTICIPANT_AUDIO", false);
    m_leaveAfterSilenceSecs = envInt("FORGEFY_LEAVE_AFTER_SILENCE_SECS", 0);

    bool ok = true;
    if (m_jwt.empty()) {
        Log::error("ZOOM_SDK_JWT is required (the orchestrator mints it)");
        ok = false;
    }
    if (m_meetingNumber.empty()) {
        Log::error("ZOOM_MEETING_NUMBER is required and must contain digits");
        ok = false;
    }
    if (m_displayName.empty()) {
        Log::error("ZOOM_DISPLAY_NAME cannot be blank — participants must be able to see the bot");
        ok = false;
    }

    return ok;
}
