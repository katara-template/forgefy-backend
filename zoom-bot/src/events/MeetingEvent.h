#ifndef FORGEFY_ZOOM_BOT_MEETINGEVENT_H
#define FORGEFY_ZOOM_BOT_MEETINGEVENT_H

#include <functional>

#include "meeting_service_interface.h"

#include "../util/Log.h"

using namespace std;
using namespace ZOOMSDK;

/**
 * Maps the SDK's meeting status enum onto the lifecycle vocabulary the
 * orchestrator understands. This is the single source of bot status — the
 * sidecar never infers state, it only relays what is emitted here.
 */
class MeetingEvent : public IMeetingServiceEvent {
public:
    MeetingEvent(function<void()> onJoined, function<void(const string&)> onTerminal)
        : m_onJoined(std::move(onJoined)), m_onTerminal(std::move(onTerminal)) {}

    void onMeetingStatusChanged(MeetingStatus status, int iResult) override {
        switch (status) {
            case MEETING_STATUS_CONNECTING:
                Status::emit("joining");
                break;

            case MEETING_STATUS_WAITINGFORHOST:
                // Host has not started the meeting yet; the SDK keeps waiting.
                Status::emit("waiting_for_host");
                break;

            case MEETING_STATUS_IN_WAITING_ROOM:
                // Admitted to the waiting room — a human must let the bot in.
                Status::emit("in_waiting_room");
                break;

            case MEETING_STATUS_INMEETING:
                Status::emit("in_meeting");
                if (!m_joinedOnce) {
                    m_joinedOnce = true;
                    m_onJoined();
                }
                break;

            case MEETING_STATUS_RECONNECTING:
                Status::emit("reconnecting");
                break;

            case MEETING_STATUS_DISCONNECTING:
                Status::emit("leaving");
                break;

            case MEETING_STATUS_ENDED:
                m_onTerminal("ended");
                break;

            case MEETING_STATUS_FAILED:
                m_onTerminal("failed: sdk result " + to_string(iResult));
                break;

            default:
                // Lock/unlock, breakout rooms, webinar promotion — no bearing
                // on whether we are capturing audio.
                break;
        }
    }

    void onMeetingParameterNotification(const MeetingParameter*) override {}
    void onMeetingStatisticsWarningNotification(StatisticsWarningType) override {}
    void onSuspendParticipantsActivities() override {
        // Host hit "Suspend Participant Activities" — recording stops with it.
        Log::error("participant activities suspended by host");
        Status::emit("suspended");
    }
    void onAICompanionActiveChangeNotice(bool) override {}
    void onMeetingTopicChanged(const zchar_t*) override {}
    void onMeetingFullToWatchLiveStream(const zchar_t*) override {}
    void onUserNetworkStatusChanged(MeetingComponentType, ConnectionQuality, unsigned int, bool) override {}

private:
    function<void()> m_onJoined;
    function<void(const string&)> m_onTerminal;
    bool m_joinedOnce = false;
};

#endif //FORGEFY_ZOOM_BOT_MEETINGEVENT_H
