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
    <Container className="mt-5" style={{ maxWidth: "400px" }}>
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
