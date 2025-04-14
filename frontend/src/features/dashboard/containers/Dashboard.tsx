import useApiCall from "hooks/useApiCall";
import { Card, Col, Container, Row } from "react-bootstrap";
import { CrimeResponse, DashboardData } from "types";
import CrimeList from "./CrimeList";

const Dashboard: React.FC = () => {

  const { data, error, loading } = useApiCall<DashboardData>("/dashboard");
  // const { data: crimeResponse, error: crimesError, loading: crimesLoading } = useApiCall<CrimeResponse>("/crime-data");


  return (
    <Container className="mt-4 scrollable-list">
      <Row className="mb-4">
        <Col md={3}>
          <Card bg="primary" text="white">
            <Card.Body>
              <Card.Title>Total Incidents</Card.Title>
              <Card.Text>{data?.total_incidents}</Card.Text>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card bg="success" text="white">
            <Card.Body>
              <Card.Title>Most Common Crime</Card.Title>
              <Card.Text>{data?.most_common_crime}</Card.Text>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card bg="warning" text="white">
            <Card.Body>
              <Card.Title>Highest Crime District</Card.Title>
              <Card.Text>{data?.highest_crime_district}</Card.Text>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card bg="danger" text="white">
            <Card.Body>
              <Card.Title>Crime Solve Rate</Card.Title>
              <Card.Text>{data?.crime_solve_rate_percent}%</Card.Text>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col>
          <CrimeList crimes={data?.recent_incidents || []} />
        </Col>
      </Row>
    </Container>
  );
};

export default Dashboard;
