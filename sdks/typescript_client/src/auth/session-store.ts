/**
 * Pluggable persistence for the signed-in session.
 *
 * The default ({@link InMemorySessionStore}) keeps the session for the life of
 * the process — fine for servers, route handlers, and tests. Web and React
 * Native apps supply a persistent store so a user stays signed in across
 * restarts; {@link persistentSessionStore} adapts both `localStorage` and
 * `AsyncStorage`. Keeping this an interface means the SDK never forces a
 * storage dependency on apps that don't want one.
 */

/** Reads, writes, and clears the serialized session string. May be sync or async. */
export interface SessionStore {
  read(): Promise<string | null> | string | null;
  write(value: string): Promise<void> | void;
  delete(): Promise<void> | void;
}

/** Non-persistent default. Lives only as long as the client. */
export class InMemorySessionStore implements SessionStore {
  private value: string | null = null;

  read(): string | null {
    return this.value;
  }

  write(value: string): void {
    this.value = value;
  }

  delete(): void {
    this.value = null;
  }
}

/**
 * The subset of `localStorage` / `@react-native-async-storage/async-storage`
 * this SDK uses. Both satisfy it (localStorage sync, AsyncStorage async).
 */
export interface KeyValueStorage {
  getItem(key: string): Promise<string | null> | string | null;
  setItem(key: string, value: string): Promise<void> | void;
  removeItem(key: string): Promise<void> | void;
}

/**
 * A persistent store backed by any {@link KeyValueStorage}.
 *
 * ```ts
 * // Next.js client component / web
 * persistentSessionStore(window.localStorage)
 * // React Native
 * import AsyncStorage from "@react-native-async-storage/async-storage";
 * persistentSessionStore(AsyncStorage)
 * ```
 */
export function persistentSessionStore(
  storage: KeyValueStorage,
  key = "forgefy.session",
): SessionStore {
  return {
    read: () => storage.getItem(key),
    write: (value) => storage.setItem(key, value),
    delete: () => storage.removeItem(key),
  };
}
