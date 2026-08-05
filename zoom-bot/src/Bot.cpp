#include "Bot.h"

#include "events/AuthEvent.h"
#include "events/MeetingEvent.h"
#include "events/RecordingEvent.h"
#include "events/ReminderEvent.h"
#include "util/Log.h"

#ifdef FORGEFY_ENABLE_CHAT_ANNOUNCE
#include "meeting_service_components/meeting_chat_interface.h"
#endif

namespace {
// One tick per second from the glib loop.
constexpr int kTickSeconds = 1;
}

bool Bot::configure() {
    if (!m_config.load()) return false;

    // Connect the audio channel before touching the SDK: if the sidecar is not
    // there, nothing downstream can work and failing now is far cheaper than
    // failing after the bot has already appeared in someone's meeting.
    if (!m_sink.start(m_config.socketPath())) {
        Status::emit("error", "could not connect to the audio sidecar");
        return false;
    }
    return true;
}

SDKError Bot::init() {
    InitParam params;

    auto host = m_config.zoomHost();
    params.strWebDomain  = host.c_str();
    params.strSupportUrl = host.c_str();
    params.emLanguageID  = LANGUAGE_English;

    // SDK logs go to the container filesystem and are captured on crash;
    // they are the only way to diagnose most join failures.
    params.enableLogByDefault  = true;
    params.enableGenerateDump  = true;

    auto err = InitSDK(params);
    if (failed(err, "initialize the SDK")) return err;

    return createServices();
}

SDKError Bot::createServices() {
    auto err = CreateMeetingService(&m_meetingService);
    if (failed(err, "create the meeting service")) return err;

    err = CreateSettingService(&m_settingService);
    if (failed(err, "create the setting service")) return err;

    auto* meetingEvent = new MeetingEvent(
        [this] { onJoined(); },
        [this](const string& reason) {
            Log::info("meeting terminal state: " + reason);
            Status::emit("ended", reason);
            // Close the PCM stream cleanly so the sidecar finalizes the
            // Deepgram connection and flushes the last utterance.
            m_sink.stop();
        });

    err = m_meetingService->SetEvent(meetingEvent);
    if (failed(err, "attach meeting events")) return err;

    return CreateAuthService(&m_authService);
}

SDKError Bot::authenticate() {
    auto err = m_authService->SetEvent(new AuthEvent([this] {
        auto e = join();
        if (failed(e, "join the meeting"))
            Status::emit("error", "join failed");
    }));
    if (failed(err, "attach auth events")) return err;

    AuthContext ctx;
    ctx.jwt_token = m_config.jwt().c_str();

    Status::emit("authenticating");
    return m_authService->SDKAuth(ctx);
}

SDKError Bot::join() {
    JoinParam joinParam;
    joinParam.userType = SDK_UT_WITHOUT_LOGIN;

    JoinParam4WithoutLogin& param = joinParam.param.withoutloginuserJoin;

    // stoull is safe here: Config guarantees digits-only and non-empty.
    param.meetingNumber = stoull(m_config.meetingNumber());
    param.userName      = m_config.displayName().c_str();
    param.psw           = m_config.password().empty() ? nullptr : m_config.password().c_str();
    param.vanityID      = nullptr;
    param.customer_key  = nullptr;
    param.webinarToken  = nullptr;

    // The bot is an observer: no camera, and its mic stays muted. Audio is
    // consumed via the raw data API, never transmitted.
    param.isVideoOff = true;
    param.isAudioOff = true;

    if (!m_config.zak().empty()) {
        param.userZAK = m_config.zak().c_str();
        Log::info("joining with a ZAK token");
    }

    if (!m_config.onBehalfToken().empty()) {
        // Since 2026-03-02 Zoom rejects Meeting SDK apps joining meetings on
        // other accounts unless they present an OBF token (or a ZAK, or use
        // RTMS). Joining our own account's meetings still works without one,
        // which is why this stays optional rather than required.
        param.onBehalfToken = m_config.onBehalfToken().c_str();
        Log::info("joining with an On-Behalf-Of token");
    } else if (m_config.zak().empty()) {
        Log::info("no OBF or ZAK token — only meetings on our own Zoom account will admit this bot");
    }

    if (!m_config.joinToken().empty()) {
        // A local-recording join token is the host's pre-authorization to
        // record, minted through the Zoom REST API. With it the bot can start
        // capture immediately instead of prompting mid-meeting.
        param.app_privilege_token = m_config.joinToken().c_str();
        Log::info("joining with a local recording token");
    }

    // Raw audio only arrives if the client actually joins the audio session.
    if (auto* audioSettings = m_settingService->GetAudioSettings())
        audioSettings->EnableAutoJoinAudio(true);

    m_started = true;
    Status::emit("joining");
    return m_meetingService->Join(joinParam);
}

void Bot::onJoined() {
    Log::success("in meeting");

    if (auto* reminderCtrl = m_meetingService->GetMeetingReminderController())
        reminderCtrl->SetEvent(new ReminderEvent());

    announce();
    requestCapture();
}

void Bot::requestCapture() {
    auto* recCtrl = m_meetingService->GetMeetingRecordingController();
    if (!recCtrl) {
        Status::emit("error", "no recording controller");
        return;
    }

    recCtrl->SetEvent(new RecordingEvent([this](bool granted) {
        if (!granted) {
            Log::error("recording privilege was not granted");
            Status::emit("consent_denied");
            return;
        }
        startCapture();
    }));

    // CanStartRawRecording() succeeding means privilege is already in hand —
    // either from a local-recording join token, or because the host granted it.
    if (recCtrl->CanStartRawRecording() == SDKERR_SUCCESS) {
        Log::success("recording privilege already granted");
        startCapture();
        return;
    }

    // NOTE ON THE CONSENT FLAG: Zoom itself refuses to release raw audio
    // without local recording privilege, so there is no "just start streaming"
    // mode to disable. The flag therefore chooses between asking the host live
    // (on) and requiring pre-authorization via join token (off) — with the flag
    // off and no token, the bot stays in the meeting capturing nothing.
    if (!m_config.requireHostConsent()) {
        Log::error("no recording privilege and live consent prompting is disabled");
        Status::emit("error", "no recording privilege (pre-authorization required)");
        return;
    }

    Log::info("asking the host for recording privilege");
    Status::emit("awaiting_consent");
    recCtrl->RequestLocalRecordingPrivilege();
}

SDKError Bot::startCapture() {
    if (m_capturing) return SDKERR_SUCCESS;

    if (m_meetingService->GetMeetingStatus() != MEETING_STATUS_INMEETING) {
        Log::error("cannot start capture before the meeting is joined");
        return SDKERR_WRONG_USAGE;
    }

    auto* recCtrl = m_meetingService->GetMeetingRecordingController();
    auto err = recCtrl->StartRawRecording();
    if (failed(err, "start raw recording")) {
        Status::emit("error", "start raw recording failed");
        return err;
    }

    // Without joining VoIP the audio session exists but carries no data.
    if (auto* audioCtrl = m_meetingService->GetMeetingAudioController()) {
        auto voipErr = audioCtrl->JoinVoip();
        failed(voipErr, "join VoIP");

        // Belt and braces: the bot must never be heard, even if a future
        // config change flips isAudioOff.
        audioCtrl->MuteAudio(0, true);
    }

    m_audioHelper = GetAudioRawdataHelper();
    if (!m_audioHelper) {
        Status::emit("error", "no raw audio helper");
        return SDKERR_UNINITIALIZE;
    }

    if (!m_audioSink)
        m_audioSink = new AudioDelegate(&m_sink, m_config.separateParticipants());

    err = m_audioHelper->subscribe(m_audioSink);
    if (failed(err, "subscribe to raw audio")) {
        Status::emit("error", "raw audio subscribe failed");
        return err;
    }

    m_capturing = true;
    Status::emit("recording");
    Log::success("streaming raw audio to the sidecar");
    return SDKERR_SUCCESS;
}

void Bot::announce() {
#ifdef FORGEFY_ENABLE_CHAT_ANNOUNCE
    // The chat builder API has changed shape across SDK releases, so this is
    // opt-in at build time (-DFORGEFY_ENABLE_CHAT_ANNOUNCE=ON). Verify it
    // against the meeting_chat_interface.h shipped in your SDK bundle first.
    auto* chatCtrl = m_meetingService->GetMeetingChatController();
    if (!chatCtrl) return;

    auto* builder = chatCtrl->GetChatMessageBuilder();
    if (!builder) return;

    auto* msg = builder->SetContent(
                           "This meeting is being transcribed by an automated notetaker.")
                       ->SetReceiver(0)
                       ->SetMessageType(SDKChatMessageType_To_All)
                       ->Build();
    if (msg) {
        chatCtrl->SendChatMsgTo(msg);
        Log::success("posted the transcription notice to meeting chat");
    }
    builder->Clear();
#else
    // Fallback disclosure: the display name is set from ZOOM_DISPLAY_NAME and
    // is visible to every participant in the roster for the whole meeting.
    Log::info("chat announcement disabled at build time; disclosure is via display name");
#endif
}

void Bot::tick() {
    if (!m_capturing || m_leaving) return;

    int idleLimit = m_config.leaveAfterSilenceSecs();
    if (idleLimit <= 0) return;

    auto* participants = m_meetingService->GetMeetingParticipantsController();
    if (!participants) return;

    auto* list = participants->GetParticipantsList();
    int count = list ? list->GetCount() : 0;

    // Zoom ends the meeting when the last human leaves, but a meeting left
    // open with only the bot in it would bill until the container is reaped.
    if (count <= 1) {
        m_aloneTicks += kTickSeconds;
        if (m_aloneTicks >= idleLimit) {
            Log::info("no other participants for " + to_string(m_aloneTicks) + "s — leaving");
            Status::emit("ended", "everyone left");
            leave();
        }
    } else {
        m_aloneTicks = 0;
    }
}

SDKError Bot::leave() {
    if (!m_meetingService) return SDKERR_UNINITIALIZE;
    if (m_leaving) return SDKERR_SUCCESS;
    m_leaving = true;

    if (m_meetingService->GetMeetingStatus() == MEETING_STATUS_IDLE)
        return SDKERR_WRONG_USAGE;

    Status::emit("leaving");
    return m_meetingService->Leave(LEAVE_MEETING);
}

void Bot::shutdown() {
    if (m_audioHelper) {
        m_audioHelper->unSubscribe();
        m_audioHelper = nullptr;
    }

    // Flush queued audio before tearing the SDK down so the tail of the
    // meeting still reaches Deepgram.
    m_sink.stop();

    if (m_meetingService) {
        DestroyMeetingService(m_meetingService);
        m_meetingService = nullptr;
    }
    if (m_settingService) {
        DestroySettingService(m_settingService);
        m_settingService = nullptr;
    }
    if (m_authService) {
        DestroyAuthService(m_authService);
        m_authService = nullptr;
    }

    delete m_audioSink;
    m_audioSink = nullptr;

    CleanUPSDK();
}

bool Bot::failed(SDKError e, const string& action) {
    bool isError = e != SDKERR_SUCCESS;
    if (action.empty()) return isError;

    if (isError)
        Log::error("failed to " + action + " (sdk error " + to_string(static_cast<int>(e)) + ")");
    else
        Log::success(action);

    return isError;
}
