import { ChartLine, File, LayoutDashboard, Map } from "lucide-react";
import { Nav } from "react-bootstrap";
import { useLocation } from "react-router-dom";
import "./Sidebar.css";

const Sidebar: React.FC = () => {
  const location = useLocation();

  return (
    <div className="bg-light border-end vh-100 p-3" style={{ width: "200px" }}>
      <Nav defaultActiveKey={location.pathname} className="flex-column">
        <Nav.Link 
          href="/dashboard" 
          active={location.pathname === "/dashboard"}
          className={location.pathname === "/dashboard" ? "active" : ""}
        >
          <LayoutDashboard size={18} className="me-2" /> Dashboard
        </Nav.Link>
        <Nav.Link 
          href="/crime-maps" 
          active={location.pathname === "/crime-maps"}
          className={location.pathname === "/crime-maps" ? "active" : ""}
        >
          <Map size={18} className="me-2" /> Crime Maps
        </Nav.Link>
        <Nav.Link 
          href="/reports" 
          active={location.pathname === "/reports"}
          className={location.pathname === "/reports" ? "active" : ""}
        >
          <File size={18} className="me-2" /> Reports
        </Nav.Link>
        <Nav.Link 
          href="/predictions" 
          active={location.pathname === "/predictions"}
          className={location.pathname === "/predictions" ? "active" : ""}
        >
          <ChartLine size={18} className="me-2" /> Predictions
        </Nav.Link>
      </Nav>
    </div>
  );
};

export default Sidebar;
