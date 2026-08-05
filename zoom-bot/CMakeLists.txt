cmake_minimum_required(VERSION 3.22)
project(forgefy_zoom_bot VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The Meeting SDK bundle (libmeetingsdk.so + h/) is not redistributable, so it
# is not vendored here — see README.md for how to place it in lib/zoomsdk.
set(ZOOM_SDK ${CMAKE_CURRENT_SOURCE_DIR}/lib/zoomsdk)

if(NOT EXISTS ${ZOOM_SDK}/h/zoom_sdk.h)
    message(FATAL_ERROR
        "Zoom Meeting SDK headers not found at ${ZOOM_SDK}/h.\n"
        "Download the Linux Meeting SDK from https://marketplace.zoom.us and "
        "unpack it into zoom-bot/lib/zoomsdk (see README.md).")
endif()

# Unlike Zoom's own sample we mint the SDK JWT in the orchestrator, so there is
# no jwt-cpp/picojson/CLI11/vcpkg here. Audio-only means no OpenCV either.
# glib supplies the event loop the SDK's callbacks are dispatched on.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED IMPORTED_TARGET glib-2.0)

find_package(Threads REQUIRED)

add_executable(forgefy-zoom-bot
        src/main.cpp
        src/Bot.cpp
        src/Bot.h
        src/Config.cpp
        src/Config.h
        src/PcmSink.cpp
        src/PcmSink.h
        src/AudioDelegate.cpp
        src/AudioDelegate.h
        src/events/AuthEvent.h
        src/events/MeetingEvent.h
        src/events/RecordingEvent.h
        src/events/ReminderEvent.h
        src/util/Log.h
)

target_include_directories(forgefy-zoom-bot PRIVATE
        ${ZOOM_SDK}/h
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_directories(forgefy-zoom-bot PRIVATE ${ZOOM_SDK})

target_link_libraries(forgefy-zoom-bot PRIVATE
        meetingsdk
        PkgConfig::GLIB
        Threads::Threads
)

# The SDK's shared object sits beside the binary at runtime.
set_target_properties(forgefy-zoom-bot PROPERTIES
        BUILD_RPATH "${ZOOM_SDK}"
        INSTALL_RPATH "${ZOOM_SDK}"
)

# Posting a chat notice on join needs an API whose shape varies between SDK
# releases; enable once verified against your bundle's meeting_chat_interface.h.
option(FORGEFY_ENABLE_CHAT_ANNOUNCE "Post a transcription notice to meeting chat" OFF)
if(FORGEFY_ENABLE_CHAT_ANNOUNCE)
    target_compile_definitions(forgefy-zoom-bot PRIVATE FORGEFY_ENABLE_CHAT_ANNOUNCE)
endif()
