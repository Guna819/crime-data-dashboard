import "./App.css";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import CrimeMaps from "./features/maps/containers/CrimeMaps";
import Dashboard from "./features/dashboard/containers/Dashboard";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Reports from "./features/reports/containers/Reports";
import Predictions from "./features/dashboard/containers/Predictions";
import Login from "./features/auth/containers/Login";
import Signup from "./features/auth/containers/Signup";
import ForgotPassword from "./features/auth/containers/ForgotPassword";
import useAuth from "hooks/useAuth";
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, SubTitle, PointElement, LineElement } from "chart.js";
import ResetPasswordPage from "./features/auth/containers/ResetPassword";

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, SubTitle, PointElement, LineElement);

function App() {

  const { user } = useAuth();

  return (
    <Router>
      { user && <Header /> }
      <div className="d-flex">
        {user && <Sidebar />}
        <div className="flex-grow-1 p-3">
          <Routes>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/crime-maps" element={<CrimeMaps />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/reset-password" element={<ResetPasswordPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
