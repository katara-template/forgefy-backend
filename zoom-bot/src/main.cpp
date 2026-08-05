#include <csignal>
#include <cstdlib>

#include <glib.h>

#include "Bot.h"
#include "util/Log.h"

namespace {

GMainLoop* g_loop = nullptr;

// Set from the signal handler, acted on from the glib loop. Doing the actual
// teardown here would call into the SDK from a signal context, which is not
// safe; flipping a flag is.
volatile sig_atomic_t g_stopRequested = 0;

// How long to let Zoom deliver the "leaving" handshake before we exit anyway.
constexpr int kLeaveGraceTicks = 5;
int g_leaveTicks = 0;

void onSignal(int) {
    g_stopRequested = 1;
}

/** Runs once a second for the lifetime of the process. */
gboolean onTick(gpointer) {
    auto& bot = Bot::instance();

    if (g_stopRequested) {
        if (g_leaveTicks == 0) {
            Log::info("shutdown requested — leaving the meeting");
            bot.leave();
        }
        if (++g_leaveTicks >= kLeaveGraceTicks) {
            g_main_loop_quit(g_loop);
            return FALSE;
        }
        return TRUE;
    }

    bot.tick();
    return TRUE;
}

} // namespace

int main() {
    auto& bot = Bot::instance();

    // SIGTERM is how the orchestrator asks the bot to leave (docker stop).
    signal(SIGINT,  onSignal);
    signal(SIGTERM, onSignal);
    // A dead sidecar surfaces as EPIPE on the socket write, not as a signal.
    signal(SIGPIPE, SIG_IGN);

    Status::emit("starting");

    if (!bot.configure()) {
        Log::error("configuration failed");
        return EXIT_FAILURE;
    }

    if (Bot::instance().init() != SDKERR_SUCCESS) {
        Status::emit("error", "sdk init failed");
        bot.shutdown();
        return EXIT_FAILURE;
    }

    // Everything past this point is callback-driven: SDKAuth returns
    // immediately and the join happens from the auth callback.
    if (bot.authenticate() != SDKERR_SUCCESS) {
        Status::emit("error", "sdk auth call failed");
        bot.shutdown();
        return EXIT_FAILURE;
    }

    g_loop = g_main_loop_new(nullptr, FALSE);
    g_timeout_add_seconds(1, onTick, nullptr);
    g_main_loop_run(g_loop);

    bot.shutdown();
    g_main_loop_unref(g_loop);

    Status::emit("stopped");
    return EXIT_SUCCESS;
}
