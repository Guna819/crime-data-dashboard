import useAuth from "hooks/useAuth";
import { useCallback, useState } from "react";
import { Alert, Button } from "react-bootstrap";

import { Form } from "react-bootstrap";

import { Container } from "react-bootstrap";
import { Link } from "react-router-dom";

// Signup Page
const Signup: React.FC = () => {
  const [fullname, setFullname] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const { signUp, loading } = useAuth();

  const handleSignup = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    signUp(fullname, email, password).then((res) => {
      if (res.error) {
        setError(res.error);
      }
    });
  }, [fullname, email, password, signUp]);

  return (
    <Container className="mt-5" style={{ maxWidth: "400px" }}>
      <h2>Sign Up</h2>
      <Form onSubmit={handleSignup}>
        <Form.Group className="mb-3">
          <Form.Label>Full Name</Form.Label>
          <Form.Control
            type="fullname"
            value={fullname}
            onChange={(e) => setFullname(e.target.value)}
            required
          />
        </Form.Group>
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
        <Button type="submit" variant="success" disabled={loading}>
          {loading ? "Signing up..." : "Sign Up"}
        </Button>
      </Form>
      <div className="mt-3">
        <p>Already have an account? <Link to="/login">Login</Link></p>
      </div>
    </Container>
  );
};

export default Signup;
