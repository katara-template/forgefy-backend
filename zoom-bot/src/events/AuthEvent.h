#ifndef FORGEFY_ZOOM_BOT_AUTHEVENT_H
#define FORGEFY_ZOOM_BOT_AUTHEVENT_H

#include <functional>
#include <string>

#include "auth_service_interface.h"

#include "../util/Log.h"

using namespace std;
using namespace ZOOMSDK;

/** Translates SDK auth results into a single success callback or a fatal status. */
class AuthEvent : public IAuthServiceEvent {
public:
    explicit AuthEvent(function<void()> onAuthenticated)
        : m_onAuthenticated(std::move(onAuthenticated)) {}

    void onAuthenticationReturn(AuthResult result) override {
        if (result == AUTHRET_SUCCESS) {
            Log::success("SDK authenticated");
            Status::emit("authenticated");
            m_onAuthenticated();
            return;
        }

        string reason;
        switch (result) {
            case AUTHRET_KEYORSECRETEMPTY: reason = "JWT was empty"; break;
            case AUTHRET_KEYORSECRETWRONG: reason = "JWT rejected — check the Meeting SDK app credentials"; break;
            case AUTHRET_ACCOUNTNOTSUPPORT: reason = "this Zoom account may not use the Meeting SDK"; break;
            case AUTHRET_ACCOUNTNOTENABLESDK: reason = "the Meeting SDK is not enabled on this account"; break;
            case AUTHRET_UNKNOWN: reason = "unknown auth failure"; break;
            case AUTHRET_SERVICE_BUSY: reason = "Zoom auth service busy"; break;
            case AUTHRET_NONE: reason = "auth never started"; break;
            case AUTHRET_OVERTIME: reason = "auth timed out"; break;
            case AUTHRET_NETWORKISSUE: reason = "network issue reaching Zoom"; break;
            case AUTHRET_CLIENT_INCOMPATIBLE: reason = "SDK version incompatible"; break;
            default: reason = "auth failed with code " + to_string(static_cast<int>(result)); break;
        }

        Log::error("authentication failed: " + reason);
        Status::emit("error", "auth: " + reason);
    }

    void onLoginReturnWithReason(LOGINSTATUS, IAccountInfo*, LoginFailReason) override {}
    void onLogout() override {}
    void onZoomIdentityExpired() override {}

    void onZoomAuthIdentityExpired() override {
        // The orchestrator mints a 24h JWT and meetings are far shorter, so
        // this should never fire; surface it rather than silently degrade.
        Log::error("SDK auth identity is about to expire");
        Status::emit("error", "auth identity expiring");
    }

private:
    function<void()> m_onAuthenticated;
};

#endif //FORGEFY_ZOOM_BOT_AUTHEVENT_H
