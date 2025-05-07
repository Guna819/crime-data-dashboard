import useApiCall from "hooks/useApiCall";
import { useCallback, useState } from "react";
import { Alert, Form } from "react-bootstrap";

import { Container } from "react-bootstrap";

import { Button } from "react-bootstrap";

const ForgotPassword: React.FC = () => {
    const [email, setEmail] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    
    const handlePasswordReset = useCallback(async (e: React.FormEvent) => {
      e.preventDefault();
      setError(''); // clear previous errors
      setSuccess('');
      try {
        console.log('Sending reset code to:', email);
    
        const response = await fetch(`http://localhost:8000/reset-password/request`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ email })
        });
    
        const data = await response.json();
    
        if (!response.ok) {
          const message = data?.detail || 'Something went wrong. Please try again.';
          throw new Error(message);
        }

        setSuccess('Reset code sent successfully.');
    
        console.log('Reset code sent:', data);
        console.log('Go to this link', `${window.location.origin}/reset-password?reset-token=${data.reset_token}`);
    
      } catch (err: any) {
        console.error('Error sending reset code:', err.message);
        setError(err.message || 'Unexpected error occurred.');
      }
    }, [email]);
  
    return (
      <Container className="mt-5" style={{ maxWidth: '400px' }}>
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
        <h2>Forgot Password</h2>
        <Form onSubmit={handlePasswordReset}>
          <Form.Group className="mb-3">
            <Form.Label>Email</Form.Label>
            <Form.Control
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </Form.Group>
          {error && <Alert variant="danger">{error}</Alert>}
          {success && <Alert variant="success">{success}</Alert>}
          <Button type="submit" variant="warning">Send Reset Code</Button>
        </Form>
      </Container>
    );
  };

export default ForgotPassword;