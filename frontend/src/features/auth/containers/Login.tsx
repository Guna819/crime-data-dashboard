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
          "https://img.freepik.com/free-photo/people-colorful-thermal-scan-with-celsius-degree-temperature_23-2149170124.jpg?t=st=1746593217~exp=1746596817~hmac=f88ac5ac6c52b242a41105e4b64edb108fd1052d3a60c158afc68ea136a88794&w=2000"
        }
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          zIndex: -2,
          filter: "blur(15px)",
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
