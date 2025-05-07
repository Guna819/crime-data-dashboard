import React, { useState, FormEvent, useCallback } from 'react';
import { Container, Row, Col, Form, Button, Alert } from 'react-bootstrap';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

type Status = {
  success: boolean | null;
  message: string;
};

const useQuery = () => {
  return new URLSearchParams(useLocation().search);
};

const ResetPasswordPage: React.FC = () => {
  const query = useQuery();
  const resetToken = query.get('reset-token');
  const navigate = useNavigate();

  const [newPassword, setNewPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [status, setStatus] = useState<Status>({ success: null, message: '' });

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!resetToken) {
      setStatus({ success: false, message: 'Missing reset token.' });
      return;
    }

    if (newPassword !== confirmPassword) {
      setStatus({ success: false, message: 'Passwords do not match.' });
      return;
    }

    try {
      await axios.post('http://localhost:8000/reset-password', {
        reset_token: resetToken,
        new_password: newPassword,
      });

      setStatus({ success: true, message: 'Password reset successfully.' });
      alert('Password reset successfully.');
      navigate('/login');
    } catch (error: any) {
      const message =
        error?.response?.data?.message || 'Something went wrong.';
      setStatus({ success: false, message });
    }
  };

  return (
    <Container className="mt-5">
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
      <Row className="justify-content-md-center">
        <Col md={6}>
          <h2>Reset Password</h2>
          <Form onSubmit={handleSubmit}>
            <Form.Group controlId="formNewPassword" className="mb-3">
              <Form.Label>New Password</Form.Label>
              <Form.Control
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
            </Form.Group>

            <Form.Group controlId="formConfirmPassword" className="mb-3">
              <Form.Label>Confirm Password</Form.Label>
              <Form.Control
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </Form.Group>

            <Button variant="primary" type="submit">
              Reset Password
            </Button>
          </Form>

          {status.message && (
            <Alert
              variant={status.success ? 'success' : 'danger'}
              className="mt-3"
            >
              {status.message}
            </Alert>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default ResetPasswordPage;
