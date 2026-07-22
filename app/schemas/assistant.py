"""Schemas for the global dashboard help assistant.

Distinct from ChatRequest/ChatResponse in project.py: those drive edits to a
single generated app (intent → build). This assistant guides users around the
app, can start a session, and keeps multiple named conversation threads.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AssistantChatRequest(BaseModel):
    message: str
    # The route the user is on when they ask (e.g. "/billing"), so the assistant
    # can ground answers in what they're currently looking at. Optional.
    page: str | None = None
    # Client-supplied recent turns, used ONLY for anonymous visitors (who have no
    # server-side history). Ignored for signed-in users, whose history is read
    # from their conversation thread so it can't be spoofed. Each: {"role","text"}.
    history: list[dict] | None = None
    # Which conversation thread this message belongs to (signed-in users). When
    # omitted, a new thread is created and its id is returned on the response.
    conversation_id: str | None = None
    # UI mode: "build" biases the assistant to treat the message as an app idea —
    # asking clarifying questions, then emitting a build_app action. Default chat.
    mode: str | None = None


class AssistantLink(BaseModel):
    """A deep link the UI renders as a tappable chip below the reply."""

    label: str
    to: str  # an in-app route path, e.g. "/billing" or "/sessions"


class AssistantAction(BaseModel):
    """A follow-up the UI should perform after showing the reply.

    - "none": nothing to do (pure advice).
    - "start_session": create + join a meeting session. platform/meeting_url are
      filled once known; the app only receives this when it can create outright.
    - "build_app": build a full app from `description`. The UI calls
      POST /assistant/build and navigates to the new project's page.
    - "auth_required": the visitor asked for something that needs an account;
      the UI shows the inline sign-in / register panel.
    """

    type: str = "none"  # "none" | "start_session" | "build_app" | "auth_required"
    platform: str | None = None  # "meet" | "zoom" | "teams" | "physical"
    meeting_url: str | None = None
    description: str | None = None  # app spec, for "build_app"


class AssistantChatResponse(BaseModel):
    response: str
    links: list[AssistantLink] = []
    action: AssistantAction | None = None
    # Echoes whether the request was authenticated, so the widget can keep its
    # UI in sync (e.g. dismiss the auth panel once the user signs in).
    authenticated: bool = False
    # The thread this reply was written to (signed-in users). Lets the client
    # adopt the id of a thread that was created on the fly by this message.
    conversation_id: str | None = None


class ConversationSummary(BaseModel):
    """A thread as shown in the switcher list — no messages, just its heading."""

    id: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary] = []


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[dict] = []


class AssistantBuildRequest(BaseModel):
    description: str


class AssistantBuildResponse(BaseModel):
    project_id: str
    session_id: str
