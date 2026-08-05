#ifndef FORGEFY_ZOOM_BOT_REMINDEREVENT_H
#define FORGEFY_ZOOM_BOT_REMINDEREVENT_H

#include "meeting_service_components/meeting_reminder_ctrl_interface.h"

#include "../util/Log.h"

using namespace ZOOMSDK;

/**
 * Zoom raises modal reminders (recording disclaimers, "you are joining a
 * meeting that is being recorded", terms acknowledgements) that a normal
 * client shows as a dialog. Headless, nobody clicks them — and an unanswered
 * reminder blocks the join. We accept them.
 *
 * Accepting a *recording disclaimer* on the bot's own behalf is not the same
 * as obtaining the host's consent to record: that gate is enforced separately
 * via local recording privilege (see Bot::onJoined).
 */
class ReminderEvent : public IMeetingReminderEvent {
public:
    void onReminderNotify(IMeetingReminderContent* content, IMeetingReminderHandler* handle) override {
        if (content)
            Log::info("meeting reminder type " + std::to_string(static_cast<int>(content->GetType())));
        if (handle)
            handle->Accept();
    }

    void onEnableReminderNotify(IMeetingReminderContent* content, IMeetingEnableReminderHandler* handle) override {
        if (handle)
            handle->Ignore();
    }
};

#endif //FORGEFY_ZOOM_BOT_REMINDEREVENT_H
