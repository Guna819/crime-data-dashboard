import useAuth from "hooks/useAuth";
import { useState } from "react";
import { Alert, Button } from "react-bootstrap";

import { Container, Form } from "react-bootstrap";
import { Link } from "react-router-dom";

const Login: React.FC = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const { login } = useAuth();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    login(email, password).then((res) => {
      if (res.error) {
        setError(res.error);
      }
    });
  };

  return (
    <Container className="mt-5 relative" style={{ maxWidth: "400px" }}>
      <img
        src={
          "https://img.freepik.com/free-vector/abstract-heat-map-thermal-style-background_1048-16348.jpg?t=st=1746532953~exp=1746536553~hmac=fc8ba1393a296dc8c88a00b77077287c01af8dcbfb2b688933dc0d624b53176d&w=1380"
        }
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          zIndex: -2,
          filter: "blur(40px)",
        }}
        alt="logo"
      />
      <h2>Login</h2>
      <Form onSubmit={handleLogin}>
        <Form.Group className="mb-3">
          <Form.Label>Email</Form.Label>
          <Form.Control
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </Form.Group>
        <Form.Group className="mb-3">
          <Form.Label>Password</Form.Label>
          <Form.Control
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </Form.Group>
        {error && <Alert variant="danger">{error}</Alert>}
        <Button type="submit" variant="primary" onClick={handleLogin}>
          Login
        </Button>
      </Form>
      <div className="mt-3">
        <p>
          Don't have an account? <Link to="/signup">Sign Up</Link>
        </p>
        <p>
          <Link to="/forgot-password">Forgot Password?</Link>
        </p>
      </div>
    </Container>
  );
};

export default Login;
