import { Col, Container, Row } from "react-bootstrap";
import BarChartPage from "./BarChartPage";
import ScatterChartPage from "./ScatterChartPage";
import PieChartPage from "./PieChartPage";

const Reports: React.FC = () => (
  <Container className="mt-4 scrollable-list">
    <h2>Reports Page</h2>
    <Row>
      <Col md={6}>
       <BarChartPage />
      </Col>
      <Col md={6}>
        <ScatterChartPage />
      </Col>
      <Col md={6}>
      <PieChartPage />
      </Col>
    </Row>
    
  </Container>
);

export default Reports;
