import useAuth from "hooks/useAuth";
import { LogOut } from "lucide-react";
import { useCallback } from "react";
import { Button, Container, Navbar } from "react-bootstrap";

import { useNavigate } from "react-router-dom";

const Header: React.FC = () => {
  const navigate = useNavigate();
  const { logout } = useAuth();

  const handleLogout = useCallback(() => {
    console.log("Logging out");
    logout();
    navigate("/login");
  }, [logout, navigate]);

  return (
    <Navbar bg="dark" variant="dark" expand="lg">
      <Container>
        <Navbar.Brand href="#">Crime Data Analysis Dashboard</Navbar.Brand>
        <Button variant="outline-light" onClick={handleLogout}>
          <LogOut size={16} className="me-2" /> Logout
        </Button>
      </Container>
    </Navbar>
  );
};

export default Header;
