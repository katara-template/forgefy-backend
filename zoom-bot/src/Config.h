#ifndef FORGEFY_ZOOM_BOT_CONFIG_H
#define FORGEFY_ZOOM_BOT_CONFIG_H

#include <string>

using namespace std;

/**
 * Configuration is read entirely from the environment.
 *
 * Notably the Meeting SDK JWT is *not* minted here — the orchestrator signs it
 * in Python and passes it in as ZOOM_SDK_JWT. That keeps the SDK client secret
 * out of the bot container entirely (a container that joins untrusted meetings
 * should not hold long-lived credentials) and drops jwt-cpp, picojson and the
 * whole vcpkg toolchain from this build.
 */
class Config {
public:
    /** Read the environment. Returns false and logs if anything required is missing. */
    bool load();

    const string& jwt()           const { return m_jwt; }
    const string& meetingNumber() const { return m_meetingNumber; }
    const string& password()      const { return m_password; }
    const string& displayName()   const { return m_displayName; }
    const string& joinToken()     const { return m_joinToken; }
    const string& zak()           const { return m_zak; }
    const string& onBehalfToken() const { return m_onBehalfToken; }
    const string& zoomHost()      const { return m_zoomHost; }
    const string& socketPath()    const { return m_socketPath; }

    bool requireHostConsent()     const { return m_requireHostConsent; }
    bool separateParticipants()   const { return m_separateParticipants; }
    int  leaveAfterSilenceSecs()  const { return m_leaveAfterSilenceSecs; }

private:
    string m_jwt;
    string m_meetingNumber;
    string m_password;
    string m_displayName        = "Forgefy Notetaker";
    string m_joinToken;
    string m_zak;
    // Required since 2026-03-02 to join meetings hosted on *other* Zoom
    // accounts. Minted per meeting by the orchestrator from the host's OAuth
    // grant; without it, external joins are rejected.
    string m_onBehalfToken;
    string m_zoomHost           = "https://zoom.us";
    string m_socketPath         = "/tmp/forgefy/audio.sock";

    bool m_requireHostConsent   = true;
    bool m_separateParticipants = false;
    int  m_leaveAfterSilenceSecs = 0;  // 0 disables the idle watchdog
};

#endif //FORGEFY_ZOOM_BOT_CONFIG_H
