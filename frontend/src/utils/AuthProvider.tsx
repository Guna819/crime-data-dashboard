import { createContext, useState, useEffect } from "react";
import { User } from "types/user";

type AuthContextType = {
  user: User | null;
  token: string | null;
  loading: boolean;
  saveUser: (user: User, token: string) => void;
  clearUser: () => void;
  setLoading: (loading: boolean) => void;
};

export const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  loading: false,
  saveUser: () => {},
  clearUser: () => {},
  setLoading: () => {},
});

const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  // Initialize state with values from localStorage if available
  const [user, setUser] = useState<User | null>(() => {
    const storedUser = localStorage.getItem("user");
    return storedUser ? JSON.parse(storedUser) : null;
  });

  const [token, setToken] = useState<string | null>(() => {
    return localStorage.getItem("token");
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const nonAuthRoutes = [
      "/login",
      "/signup",
      "/forgot-password",
      "/reset-password",
    ];

    const emptyPath = window.location.pathname === '/';

    if (
      nonAuthRoutes.includes(window.location.pathname) || emptyPath) {
      if (user) {
        window.location.href = "/dashboard";
      }
      if (!emptyPath) {
        return;
      }
    }
    if (!user) {
      window.location.href = "/login";
    }
  }, [user, setUser]);

  // Function to save user and token to both state and localStorage
  const saveUser = (user: User, token: string) => {
    setUser(user);
    setToken(token);
    localStorage.setItem("user", JSON.stringify(user)); // Save user object as a string
    localStorage.setItem("token", token); // Save token
    window.location.href = "/dashboard";
  };

  // Function to clear user and token from both state and localStorage
  const clearUser = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem("user"); // Remove user from localStorage
    localStorage.removeItem("token"); // Remove token from localStorage
  };

  // Persist state changes to localStorage (for token or user updates)
  useEffect(() => {
    if (user && token) {
      localStorage.setItem("user", JSON.stringify(user));
      localStorage.setItem("token", token);
    }
  }, [user, token]);

  return (
    <AuthContext.Provider
      value={{ user, token, saveUser, clearUser, loading, setLoading }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;
