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
