import { useCallback, useContext, useEffect } from "react";
import { AuthResponse, LoginResponse, SignupResponse } from "types";
import { AuthContext } from "utils/AuthProvider";

const useAuth = () => {
  const {
    user,
    token,
    saveUser,
    clearUser,
    loading,
    setLoading,
  } = useContext(AuthContext);

  const signUp = async (
    fullname: string,
    email: string,
    password: string
  ): Promise<SignupResponse> => {
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/signup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ fullname, email, password }),
      });
      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        throw new Error(data.detail || "Signup failed");
      }
      console.log("Signup successful", data);
      saveUser(data.user, data.access_token);
      return { access_token: data.access_token, message: data.message }; // Assuming a token is returned
    } catch (err: any) {
      setLoading(false);
      return { error: err.message };
    }
  };

  // Function to handle login
  const login = async (
    email: string,
    password: string
  ): Promise<LoginResponse> => {
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        throw new Error(data.detail || "Login failed");
      }
      console.log("Login successful", data);
      saveUser(data.user, data.access_token);
      return { access_token: data.access_token, message: data.message, user: data.user }; // Assuming a token is returned
    } catch (err: any) {
      setLoading(false);
      return { error: err.message };
    }
  };

  const logout = useCallback(() => {
    clearUser();
  }, [clearUser]);

  return {
    user,
    token,
    login,
    logout,
    loading,
    signUp,
  };
};

export default useAuth;
