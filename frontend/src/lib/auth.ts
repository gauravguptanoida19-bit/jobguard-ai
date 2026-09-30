export interface UserProfile {
  id: string;
  email: string;
  name: string;
  role: string;
  organization: string;
}

export interface AuthSession {
  accessToken: string;
  user: UserProfile;
}

const TOKEN_KEY = "jobguard_access_token";
const USER_KEY = "jobguard_user_profile";
export const AUTH_EVENT = "jobguard-auth-change";

export function getStoredUser(): UserProfile | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function storeAuth(accessToken: string, user: UserProfile): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(TOKEN_KEY, accessToken);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  window.dispatchEvent(new Event(AUTH_EVENT));
}

export function clearAuth(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  window.dispatchEvent(new Event(AUTH_EVENT));
}
