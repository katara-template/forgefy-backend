#ifndef FORGEFY_ZOOM_BOT_RECORDINGEVENT_H
#define FORGEFY_ZOOM_BOT_RECORDINGEVENT_H

#include <functional>

#include "meeting_service_components/meeting_recording_interface.h"

#include "../util/Log.h"

using namespace std;
using namespace ZOOMSDK;

/**
 * Carries the host's answer to our local-recording request.
 *
 * This is the consent gate. Zoom will not hand over raw audio until local
 * recording privilege is granted, so a host who never approves means the bot
 * sits in the meeting transcribing nothing — which is the behaviour we want.
 */
class RecordingEvent : public IMeetingRecordingCtrlEvent {
public:
    explicit RecordingEvent(function<void(bool)> onPrivilegeChanged)
        : m_onPrivilegeChanged(std::move(onPrivilegeChanged)) {}

    void onRecordPrivilegeChanged(bool bCanRec) override {
        Log::info(string("recording privilege ") + (bCanRec ? "granted" : "revoked"));
        m_onPrivilegeChanged(bCanRec);
    }

    void onLocalRecordingPrivilegeRequestStatus(RequestLocalRecordingStatus status) override {
        switch (status) {
            case AttendeeLocalRecording_Request_Granted:
                Status::emit("consent_granted");
                break;
            case AttendeeLocalRecording_Request_Denied:
                Log::error("host denied the recording request");
                Status::emit("consent_denied");
                break;
            case AttendeeLocalRecording_Request_Timeout:
                Log::error("recording request timed out");
                Status::emit("consent_timeout");
                break;
            default:
                break;
        }
    }

    void onRecordingStatus(RecordingStatus) override {}
    void onCloudRecordingStatus(RecordingStatus) override {}
    void onLocalRecordingPrivilegeRequested(IRequestLocalRecordingPrivilegeHandler*) override {}
    void onCloudRecordingStorageFull(time_t) override {}
    void onRequestCloudRecordingResponse(RequestStartCloudRecordingStatus) override {}
    void onStartCloudRecordingRequested(IRequestStartCloudRecordingHandler*) override {}
    void onEnableAndStartSmartRecordingRequested(IRequestEnableAndStartSmartRecordingHandler*) override {}
    void onSmartRecordingEnableActionCallback(ISmartRecordingEnableActionHandler*) override {}
    void onTranscodingStatusChanged(TranscodingStatus, const zchar_t*) override {}

private:
    function<void(bool)> m_onPrivilegeChanged;
};

#endif //FORGEFY_ZOOM_BOT_RECORDINGEVENT_H
