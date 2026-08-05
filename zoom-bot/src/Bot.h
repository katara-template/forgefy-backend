#ifndef FORGEFY_ZOOM_BOT_BOT_H
#define FORGEFY_ZOOM_BOT_BOT_H

#include <string>

#include "zoom_sdk.h"
#include "rawdata/zoom_rawdata_api.h"
#include "meeting_service_interface.h"
#include "auth_service_interface.h"
#include "setting_service_interface.h"
#include "meeting_service_components/meeting_audio_interface.h"
#include "meeting_service_components/meeting_participants_ctrl_interface.h"
#include "meeting_service_components/meeting_recording_interface.h"

#include "Config.h"
#include "PcmSink.h"
#include "AudioDelegate.h"

using namespace std;
using namespace ZOOMSDK;

/**
 * One bot, one meeting, one process.
 *
 * Lifecycle: init → auth → join → (host consent) → raw capture → leave.
 * Every transition is published on stdout by the event handlers; this class
 * owns only the SDK objects and the ordering between them.
 */
class Bot {
public:
    static Bot& instance() {
        static Bot bot;
        return bot;
    }

    bool configure();
    SDKError init();
    SDKError authenticate();

    /** Called on the glib loop; drives the idle-participant watchdog. */
    void tick();

    SDKError leave();
    void shutdown();

    bool hasStarted() const { return m_started; }

private:
    Bot() = default;
    Bot(const Bot&) = delete;
    Bot& operator=(const Bot&) = delete;

    SDKError createServices();
    SDKError join();

    /** Fired once the SDK reports MEETING_STATUS_INMEETING. */
    void onJoined();

    /** Request or verify local recording privilege, then begin capture. */
    void requestCapture();

    /** Subscribe to raw audio. Only legal once privilege is granted. */
    SDKError startCapture();

    /** Post the "this meeting is being transcribed" notice into meeting chat. */
    void announce();

    static bool failed(SDKError e, const string& action = "");

    Config m_config;
    PcmSink m_sink;

    IMeetingService* m_meetingService = nullptr;
    ISettingService* m_settingService = nullptr;
    IAuthService*    m_authService    = nullptr;

    IZoomSDKAudioRawDataHelper* m_audioHelper = nullptr;
    AudioDelegate*              m_audioSink   = nullptr;

    bool m_started        = false;
    bool m_capturing      = false;
    bool m_leaving        = false;
    int  m_aloneTicks     = 0;
};

#endif //FORGEFY_ZOOM_BOT_BOT_H
